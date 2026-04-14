"""
Computerized Adaptive Testing (CAT) Engine - v7.0 Core Algorithm

⚠️  SCRIPT-ONLY — Not part of the production server pipeline.
    Only referenced by scripts/validate_phase2.py for offline CAT validation.
    The production orchestrator uses CAT logic embedded in orchestrator.py directly.

Implements Item Response Theory-based adaptive testing for optimal
test efficiency. Dynamically selects items to maximize measurement
precision while minimizing test length.

Reference:
- Weiss, D. J. (1982). Improving measurement quality and efficiency with adaptive testing.
- van der Linden, W. J., & Glas, C. A. (2010). Elements of Adaptive Testing.

Key Features:
- Maximum Fisher Information item selection
- Maximum Likelihood Estimation (MLE) for ability updates
- Content balancing constraints
- Early stopping rules based on measurement precision
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from app.core.logging import get_logger
from .irt_calibration import IRTParameters, get_calibrated_params

logger = get_logger(__name__)


class StoppingRule(Enum):
    """Stopping criteria for adaptive testing."""
    FIXED_LENGTH = "fixed_length"
    STANDARD_ERROR = "standard_error"
    INFORMATION_GAIN = "information_gain"
    CONFIDENCE_INTERVAL = "confidence_interval"


@dataclass
class AbilityEstimate:
    """Current ability estimate with uncertainty."""
    theta: float
    se: float
    n_items: int
    information: float
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta": round(self.theta, 3),
            "se": round(self.se, 3),
            "n_items": self.n_items,
            "information": round(self.information, 3),
            "ci_95": [round(self.confidence_interval[0], 3), 
                     round(self.confidence_interval[1], 3)],
        }


@dataclass
class TestItem:
    """An item in the adaptive test pool."""
    case_id: str
    dimension: str
    irt_params: Optional[IRTParameters] = None
    content_tags: List[str] = field(default_factory=list)
    exposure_count: int = 0
    last_selected: Optional[str] = None
    
    def information_at(self, theta: float) -> float:
        """Calculate Fisher information at ability theta."""
        if self.irt_params:
            info = self.irt_params.calculate_information(theta)
            return float(info) if isinstance(info, (int, float)) else float(info[0])
        # Fallback: standard normal approximation
        return np.exp(-(theta ** 2) / 2) / np.sqrt(2 * np.pi)
    
    def probability_correct(self, theta: float) -> float:
        """Probability of correct response at theta."""
        if self.irt_params:
            return self.irt_params.probability_correct(theta)
        return 0.5  # Default guess


@dataclass
class ItemResponse:
    """Response to a test item."""
    item: TestItem
    correct: bool
    response_time_ms: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class CATSession:
    """Complete adaptive testing session."""
    session_id: str
    ability_estimate: AbilityEstimate
    responses: List[ItemResponse]
    items_administered: List[str]
    stopping_rule_used: StoppingRule
    test_length: int
    target_length: int
    total_information: float
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ability": self.ability_estimate.to_dict(),
            "test_length": self.test_length,
            "target_length": self.target_length,
            "stopping_rule": self.stopping_rule_used.value,
            "total_information": round(self.total_information, 3),
            "items": self.items_administered,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class CATengine:
    """
    Computerized Adaptive Testing Engine.
    
    Reduces test length by 30-50% while maintaining or improving precision
    compared to fixed-length tests.
    
    Algorithm:
    1. Initialize ability estimate (theta = 0.0)
    2. Select item with maximum information at current theta
    3. Administer item and record response
    4. Update theta estimate using MLE
    5. Check stopping criteria
    6. Repeat 2-5 until stopping criteria met
    """
    
    # Default configuration
    DEFAULT_TARGET_SE = 0.3  # Target standard error (~90% CI width)
    DEFAULT_MAX_ITEMS = 30
    DEFAULT_MIN_ITEMS = 5
    CONTENT_BALANCE_PENALTY = 0.1
    
    def __init__(
        self,
        item_pool: List[TestItem],
        target_se: float = DEFAULT_TARGET_SE,
        max_items: int = DEFAULT_MAX_ITEMS,
        min_items: int = DEFAULT_MIN_ITEMS,
        content_constraints: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Initialize CAT engine.
        
        Args:
            item_pool: Available test items with IRT parameters
            target_se: Target standard error for stopping
            max_items: Maximum test length
            min_items: Minimum test length before early stopping allowed
            content_constraints: Dict mapping content area to (min, max) counts
        """
        self.item_pool = item_pool
        self.target_se = target_se
        self.max_items = max_items
        self.min_items = min_items
        self.content_constraints = content_constraints or {}
        
        # Track item exposure for content balancing
        self.dimension_counts: Dict[str, int] = {}
        
        logger.info(
            f"CAT Engine initialized: {len(item_pool)} items, "
            f"target_se={target_se}, max_items={max_items}"
        )
    
    def start_session(
        self,
        session_id: str,
        initial_theta: float = 0.0
    ) -> CATSession:
        """
        Start a new adaptive testing session.
        
        Args:
            session_id: Unique session identifier
            initial_theta: Starting ability estimate (default: 0.0 = average)
            
        Returns:
            CATSession initialized with first ability estimate
        """
        initial_estimate = AbilityEstimate(
            theta=initial_theta,
            se=1.0,  # High uncertainty initially
            n_items=0,
            information=0.0,
            confidence_interval=(initial_theta - 2.0, initial_theta + 2.0)
        )
        
        session = CATSession(
            session_id=session_id,
            ability_estimate=initial_estimate,
            responses=[],
            items_administered=[],
            stopping_rule_used=StoppingRule.FIXED_LENGTH,
            test_length=0,
            target_length=self.max_items,
            total_information=0.0
        )
        
        return session
    
    def select_next_item(
        self,
        current_estimate: AbilityEstimate,
        available_items: List[TestItem],
        exclude_items: List[str]
    ) -> Optional[TestItem]:
        """
        Select optimal next item using Maximum Fisher Information criterion.
        
        With content balancing: penalizes over-represented content areas.
        
        Args:
            current_estimate: Current ability estimate
            available_items: Pool of items to select from
            exclude_items: Items already administered (to avoid)
            
        Returns:
            Selected TestItem or None if no suitable items
        """
        theta = current_estimate.theta
        best_item = None
        max_utility = -float('inf')
        
        for item in available_items:
            # Skip already administered items
            if item.case_id in exclude_items:
                continue
            
            # Calculate information at current theta
            info = item.information_at(theta)
            
            # Content balancing penalty
            dim_count = self.dimension_counts.get(item.dimension, 0)
            content_penalty = self.CONTENT_BALANCE_PENALTY * dim_count
            
            # Check content constraints
            constraint_penalty = 0.0
            for tag in item.content_tags:
                if tag in self.content_constraints:
                    min_req, max_req = self.content_constraints[tag]
                    current_count = sum(
                        1 for r in self.responses 
                        if tag in r.item.content_tags
                    ) if hasattr(self, 'responses') else 0
                    
                    if current_count >= max_req:
                        constraint_penalty += 1.0  # Heavy penalty for exceeding max
                    elif current_count < min_req:
                        info *= 1.1  # Boost for under-represented required content
            
            # Combined utility
            utility = info - content_penalty - constraint_penalty
            
            if utility > max_utility:
                max_utility = utility
                best_item = item
        
        if best_item:
            logger.debug(
                f"Selected item {best_item.case_id} (dim={best_item.dimension}) "
                f"with utility={max_utility:.3f}"
            )
        
        return best_item
    
    def update_ability_estimate(
        self,
        responses: List[ItemResponse]
    ) -> AbilityEstimate:
        """
        Update ability estimate using Maximum Likelihood Estimation.
        
        Finds theta that maximizes the likelihood of observed responses.
        Uses standard numerical optimization (Brent's method).
        
        Args:
            responses: List of item responses
            
        Returns:
            Updated AbilityEstimate
        """
        if not responses:
            return AbilityEstimate(
                theta=0.0, se=1.0, n_items=0, 
                information=0.0, confidence_interval=(-2.0, 2.0)
            )
        
        def neg_log_likelihood(theta: float) -> float:
            """Calculate negative log-likelihood for optimization."""
            log_l = 0.0
            for resp in responses:
                p = resp.item.probability_correct(theta)
                p = max(1e-10, min(1 - 1e-10, p))  # Avoid log(0)
                
                if resp.correct:
                    log_l += math.log(p)
                else:
                    log_l += math.log(1 - p)
            
            return -log_l
        
        # Optimize using bounded Brent's method
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(-4.0, 4.0),
            method='bounded'
        )
        
        theta_mle = result.x
        
        # Calculate standard error using Fisher information
        total_info = 0.0
        for resp in responses:
            info = resp.item.information_at(theta_mle)
            total_info += info
        
        # SE = 1 / sqrt(I(theta))
        if total_info > 0.001:
            se = 1.0 / math.sqrt(total_info)
        else:
            se = 1.0  # Default high uncertainty
        
        # 95% confidence interval
        ci_lower = theta_mle - 1.96 * se
        ci_upper = theta_mle + 1.96 * se
        
        estimate = AbilityEstimate(
            theta=round(theta_mle, 3),
            se=round(se, 3),
            n_items=len(responses),
            information=round(total_info, 3),
            confidence_interval=(round(ci_lower, 3), round(ci_upper, 3))
        )
        
        logger.debug(
            f"Ability estimate updated: θ={estimate.theta}, SE={estimate.se}"
        )
        
        return estimate
    
    def check_stopping_criteria(
        self,
        current_estimate: AbilityEstimate,
        n_items: int
    ) -> Tuple[bool, StoppingRule]:
        """
        Check if stopping criteria are met.
        
        Stopping rules (in order of priority):
        1. Max items reached
        2. Min items not yet reached
        3. Standard error below target
        4. Information gain below threshold
        
        Args:
            current_estimate: Current ability estimate
            n_items: Number of items administered
            
        Returns:
            Tuple of (should_stop, rule_used)
        """
        # Rule 1: Maximum items
        if n_items >= self.max_items:
            return True, StoppingRule.FIXED_LENGTH
        
        # Rule 2: Minimum items not reached
        if n_items < self.min_items:
            return False, StoppingRule.FIXED_LENGTH
        
        # Rule 3: Target precision reached
        if current_estimate.se <= self.target_se:
            return True, StoppingRule.STANDARD_ERROR
        
        # Rule 4: Confidence interval width
        ci_width = (current_estimate.confidence_interval[1] - 
                   current_estimate.confidence_interval[0])
        if ci_width <= self.target_se * 3.92:  # 95% CI
            return True, StoppingRule.CONFIDENCE_INTERVAL
        
        return False, StoppingRule.FIXED_LENGTH
    
    def run_adaptive_test(
        self,
        session_id: str,
        response_generator: Any,  # Callable that takes item and returns response
        initial_theta: float = 0.0
    ) -> CATSession:
        """
        Run complete adaptive testing session.
        
        Args:
            session_id: Unique session identifier
            response_generator: Function(item) -> correct (bool)
            initial_theta: Starting ability estimate
            
        Returns:
            Complete CATSession with results
        """
        session = self.start_session(session_id, initial_theta)
        available_items = self.item_pool.copy()
        
        logger.info(f"Starting adaptive test session {session_id}")
        
        while True:
            # Check stopping criteria
            should_stop, rule = self.check_stopping_criteria(
                session.ability_estimate,
                len(session.responses)
            )
            
            if should_stop:
                session.stopping_rule_used = rule
                logger.info(
                    f"Test stopped after {len(session.responses)} items "
                    f"using rule: {rule.value}"
                )
                break
            
            # Select next item
            item = self.select_next_item(
                session.ability_estimate,
                available_items,
                session.items_administered
            )
            
            if not item:
                logger.warning("No suitable items available")
                break
            
            # "Administer" item (via response generator)
            try:
                correct = response_generator(item)
            except Exception as e:
                logger.error(f"Response generation failed for {item.case_id}: {e}")
                correct = False
            
            response = ItemResponse(
                item=item,
                correct=correct
            )
            
            # Record response
            session.responses.append(response)
            session.items_administered.append(item.case_id)
            session.test_length += 1
            
            # Update dimension count
            self.dimension_counts[item.dimension] = \
                self.dimension_counts.get(item.dimension, 0) + 1
            
            # Update ability estimate
            session.ability_estimate = self.update_ability_estimate(
                session.responses
            )
            session.total_information += item.information_at(
                session.ability_estimate.theta
            )
            
            logger.debug(
                f"Item {item.case_id}: {'correct' if correct else 'incorrect'}, "
                f"θ={session.ability_estimate.theta}"
            )
        
        session.end_time = datetime.utcnow().isoformat()
        
        logger.info(
            f"Session {session_id} complete: "
            f"final θ={session.ability_estimate.theta}, "
            f"SE={session.ability_estimate.se}, "
            f"n={session.test_length}"
        )
        
        return session
    
    def compare_efficiency(
        self,
        session: CATSession,
        fixed_length_reference: int = 30
    ) -> Dict[str, Any]:
        """
        Compare CAT efficiency vs fixed-length test.
        
        Args:
            session: Completed CAT session
            fixed_length_reference: Reference fixed test length
            
        Returns:
            Efficiency metrics
        """
        actual_length = session.test_length
        reduction = (fixed_length_reference - actual_length) / fixed_length_reference
        
        return {
            "cat_length": actual_length,
            "fixed_length": fixed_length_reference,
            "reduction": round(reduction, 2),
            "reduction_percent": round(reduction * 100, 1),
            "precision_ratio": round(
                (1.0 / session.ability_estimate.se) / 
                math.sqrt(fixed_length_reference), 2
            ),
            "efficiency_score": round(
                reduction * (1.0 / session.ability_estimate.se), 2
            ),
        }


def create_item_pool_from_cases(
    cases: List[Any],
    irt_cache_path: Optional[str] = None
) -> List[TestItem]:
    """
    Create CAT item pool from test cases.
    
    Args:
        cases: List of test case objects
        irt_cache_path: Optional path to IRT calibration cache
        
    Returns:
        List of TestItems with IRT parameters
    """
    items = []
    
    # Load IRT calibration if available
    if irt_cache_path:
        from .irt_calibration import IRTCalibrator
        calibrator = IRTCalibrator()
        calibration = calibrator.load_calibration_cache(irt_cache_path)
    else:
        calibration = {}
    
    for case in cases:
        case_id = getattr(case, 'id', str(case))
        dimension = getattr(case, 'dimension', 'unknown')
        
        # Get IRT parameters if available
        if case_id in calibration:
            irt_params = calibration[case_id].parameters
        else:
            irt_params = None
        
        item = TestItem(
            case_id=case_id,
            dimension=dimension,
            irt_params=irt_params,
            content_tags=getattr(case, 'content_tags', [])
        )
        items.append(item)
    
    logger.info(f"Created item pool with {len(items)} items")
    return items


# Convenience function for quick testing
def run_demo_cat():
    """Run a demo adaptive test with synthetic items."""
    import random
    
    # Create synthetic item pool
    items = []
    for i in range(50):
        # Create items with varying difficulty
        b = random.uniform(-2, 2)
        a = random.uniform(0.8, 1.5)
        
        irt_params = IRTParameters(
            a=a, b=b, c=0.0,
            n_calibrated=100,
            calibration_date="2026-04-11",
            data_source="demo"
        )
        
        item = TestItem(
            case_id=f"item_{i:03d}",
            dimension=random.choice(["reasoning", "coding", "instruction"]),
            irt_params=irt_params
        )
        items.append(item)
    
    # Create engine
    engine = CATengine(
        items,
        target_se=0.3,
        max_items=20,
        min_items=5
    )
    
    # Simulated response generator (theta = 1.0)
    true_theta = 1.0
    
    def response_generator(item: TestItem) -> bool:
        p = item.probability_correct(true_theta)
        return random.random() < p
    
    # Run test
    session = engine.run_adaptive_test("demo_001", response_generator)
    
    print(f"\nDemo CAT Results:")
    print(f"True ability: {true_theta}")
    print(f"Estimated: {session.ability_estimate.theta}")
    print(f"Standard Error: {session.ability_estimate.se}")
    print(f"Items administered: {session.test_length}")
    print(f"Stopping rule: {session.stopping_rule_used.value}")
    
    efficiency = engine.compare_efficiency(session, fixed_length_reference=30)
    print(f"Efficiency: {efficiency['reduction_percent']}% reduction vs fixed test")
    
    return session


if __name__ == "__main__":
    run_demo_cat()
