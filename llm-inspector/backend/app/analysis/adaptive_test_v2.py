"""Adaptive Testing 2.0 - Enhanced CAT Engine for LLM Inspector v8.0.

Implements v8.0 improvements:
- Multi-dimensional adaptive testing
- Token-cost aware item selection
- Real-time ability estimation updates
- Smart stopping rules with confidence intervals

Reference:
- van der Linden (2010): Elements of Adaptive Testing
- Mulder & van der Linden (2009): Multidimensional Adaptive Testing
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
from datetime import datetime, timezone

from app.analysis.irt_params import IRTParameters, IRTParameterDB
from app.analysis.theta_scoring import ThetaScoringEngine, ThetaScore
from app.core.schemas import TestCase, TestCaseV8
from app.core.logging import get_logger

logger = get_logger(__name__)


class StoppingRule(Enum):
    """Stopping criteria for adaptive testing."""
    FIXED_LENGTH = "fixed_length"           # Fixed number of items
    STANDARD_ERROR = "standard_error"       # SE < threshold
    INFORMATION = "information"            # Total information > threshold
    CONFIDENCE_WIDTH = "confidence_width"   # CI width < threshold
    TOKEN_BUDGET = "token_budget"           # Token limit reached


@dataclass
class AdaptiveTestConfig:
    """Configuration for adaptive testing session."""
    
    # Test structure
    dimensions: List[str] = field(default_factory=list)
    min_items: int = 5
    max_items: int = 20
    items_per_dimension: int = 5
    
    # Stopping rules
    stopping_rules: List[StoppingRule] = field(
        default_factory=lambda: [
            StoppingRule.FIXED_LENGTH,
            StoppingRule.STANDARD_ERROR
        ]
    )
    target_se: float = 0.3
    target_information: float = 5.0
    max_confidence_width: float = 1.0  # Theta units
    
    # Cost constraints
    token_budget: int = 10000
    
    # Content constraints
    min_items_per_dimension: int = 2
    max_exposure_rate: float = 0.5  # Prevent over-exposure of items
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "min_items": self.min_items,
            "max_items": self.max_items,
            "target_se": self.target_se,
            "token_budget": self.token_budget,
        }


@dataclass
class TestSessionState:
    """Current state of an adaptive test session."""
    
    session_id: str
    config: AdaptiveTestConfig
    
    # Administered items
    administered_items: List[str] = field(default_factory=list)
    responses: Dict[str, bool] = field(default_factory=dict)  # item_id -> correct
    
    # Ability estimates per dimension
    ability_estimates: Dict[str, ThetaScore] = field(default_factory=dict)
    
    # Token tracking
    tokens_used: int = 0
    
    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: Optional[datetime] = None
    
    # Status
    is_complete: bool = False
    completion_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "dimensions": self.config.dimensions,
            "items_administered": len(self.administered_items),
            "tokens_used": self.tokens_used,
            "is_complete": self.is_complete,
            "ability_estimates": {
                dim: est.to_dict() 
                for dim, est in self.ability_estimates.items()
            },
        }


class AdaptiveTestSelector:
    """
    Enhanced adaptive test item selector.
    
    Features:
    - Maximum information selection
    - Token-cost optimization
    - Multi-dimensional content balancing
    - Exposure control
    """
    
    def __init__(self, irt_db: Optional[IRTParameterDB] = None):
        """
        Initialize selector.
        
        Args:
            irt_db: IRT parameter database (creates default if None)
        """
        if irt_db is None:
            irt_db = IRTParameterDB()
        self.irt_db = irt_db
        
        # Statistics
        self.stats = {
            "selections": 0,
            "cache_hits": 0,
        }
        
        logger.info("AdaptiveTestSelector v2.0 initialized")
    
    def select_next_item(
        self,
        session: TestSessionState,
        available_items: List[TestCaseV8],
    ) -> Optional[TestCaseV8]:
        """
        Select next item using maximum information criterion.
        
        Algorithm:
        1. Filter out administered items
        2. For each dimension, calculate information at current theta
        3. Apply content balancing constraints
        4. Select item with max adjusted information
        
        Args:
            session: Current test session
            available_items: Pool of available items
            
        Returns:
            Selected item or None if no suitable item
        """
        # Filter out already administered items
        available = [
            item for item in available_items 
            if item.id not in session.administered_items
        ]
        
        if not available:
            return None
        
        # Calculate information for each item
        item_scores = []
        
        for item in available:
            # Get IRT parameters
            params_result = self.irt_db.get_parameters(item.id)
            if not params_result:
                continue
            
            params, _ = params_result
            
            # Skip invalid parameters
            if not params.is_valid:
                continue
            
            # Get current theta estimate for item's dimension
            dimension = item.dimension or "general"
            if dimension in session.ability_estimates:
                current_theta = session.ability_estimates[dimension].theta
            else:
                current_theta = 0.0  # Default prior
            
            # Calculate Fisher information
            info = params.calculate_information(current_theta)
            
            # Apply token cost adjustment
            estimated_tokens = self._estimate_tokens(item)
            cost_factor = 1.0 / (1 + estimated_tokens / 1000)
            
            # Apply dimension balance adjustment
            dim_count = sum(
                1 for i in session.administered_items 
                if self._get_item_dimension(i) == dimension
            )
            balance_factor = 1.0 / (1 + dim_count / session.config.items_per_dimension)
            
            # Final score
            adjusted_info = info * cost_factor * balance_factor
            
            item_scores.append((item, adjusted_info, info))
        
        if not item_scores:
            return None
        
        # Sort by adjusted information (descending)
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top item
        selected = item_scores[0][0]
        self.stats["selections"] += 1
        
        logger.debug(f"Selected item {selected.id} with info={item_scores[0][2]:.3f}")
        
        return selected
    
    def _estimate_tokens(self, item: TestCaseV8) -> int:
        """Estimate token cost for an item."""
        base_tokens = 500  # Prompt overhead
        
        # Add prompt length estimate
        if item.user_prompt:
            base_tokens += len(item.user_prompt) // 4
        
        # Add max response tokens
        base_tokens += item.max_tokens
        
        return base_tokens
    
    def _get_item_dimension(self, item_id: str) -> str:
        """Get dimension for an item ID (would query database in production)."""
        # Simple heuristic: extract dimension from ID prefix
        if "_" in item_id:
            return item_id.split("_")[0]
        return "general"
    
    def select_item_for_dimension(
        self,
        dimension: str,
        current_theta: float,
        exclude_items: Set[str],
        max_tokens: Optional[int] = None,
    ) -> Optional[Tuple[str, IRTParameters]]:
        """
        Select optimal item for a specific dimension.
        
        Args:
            dimension: Target dimension
            current_theta: Current ability estimate
            exclude_items: Items to exclude (already used)
            max_tokens: Optional token budget constraint
            
        Returns:
            Tuple of (item_id, params) or None
        """
        # Get all items for dimension
        params_list = self.irt_db.get_parameters_by_dimension(dimension)
        
        # Filter valid and unused items
        candidates = [
            (p.case_id, p) for p in params_list
            if p.is_valid and p.case_id not in exclude_items
        ]
        
        if not candidates:
            return None
        
        # Calculate information for each
        item_info = [
            (item_id, params, params.calculate_information(current_theta))
            for item_id, params in candidates
        ]
        
        # Sort by information (descending)
        item_info.sort(key=lambda x: x[2], reverse=True)
        
        return (item_info[0][0], item_info[0][1]) if item_info else None


class AdaptiveTestEngine:
    """
    Adaptive Testing 2.0 Engine.
    
    Manages complete adaptive test sessions with:
    - Real-time ability estimation
    - Smart stopping rules
    - Multi-dimensional scoring
    """
    
    def __init__(
        self,
        selector: Optional[AdaptiveTestSelector] = None,
        scoring_engine: Optional[ThetaScoringEngine] = None,
    ):
        """
        Initialize engine.
        
        Args:
            selector: Item selector (creates default if None)
            scoring_engine: Scoring engine (creates default if None)
        """
        self.selector = selector or AdaptiveTestSelector()
        self.scoring = scoring_engine or ThetaScoringEngine("mle")
        
        # Active sessions
        self.sessions: Dict[str, TestSessionState] = {}
        
        logger.info("AdaptiveTestEngine v2.0 initialized")
    
    def start_session(
        self,
        config: AdaptiveTestConfig,
        session_id: Optional[str] = None,
    ) -> TestSessionState:
        """
        Start new adaptive test session.
        
        Args:
            config: Test configuration
            session_id: Optional session ID (generated if None)
            
        Returns:
            New session state
        """
        if session_id is None:
            session_id = f"cat_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{id(self)}"
        
        session = TestSessionState(
            session_id=session_id,
            config=config,
        )
        
        self.sessions[session_id] = session
        logger.info(f"Started adaptive test session {session_id}")
        
        return session
    
    def get_next_item(
        self,
        session_id: str,
        available_items: List[TestCaseV8],
    ) -> Optional[TestCaseV8]:
        """
        Get next item for session.
        
        Args:
            session_id: Session identifier
            available_items: Available item pool
            
        Returns:
            Next item or None if test complete
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.is_complete:
            return None
        
        # Check stopping rules
        if self._should_stop(session):
            session.is_complete = True
            return None
        
        # Select next item
        item = self.selector.select_next_item(session, available_items)
        
        return item
    
    def record_response(
        self,
        session_id: str,
        item_id: str,
        correct: bool,
        tokens_used: int = 0,
    ) -> TestSessionState:
        """
        Record response and update ability estimates.
        
        Args:
            session_id: Session identifier
            item_id: Item that was administered
            correct: Whether response was correct
            tokens_used: Tokens consumed by this item
            
        Returns:
            Updated session state
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Record response
        session.administered_items.append(item_id)
        session.responses[item_id] = correct
        session.tokens_used += tokens_used
        session.last_update = datetime.now(timezone.utc)
        
        # Update ability estimates
        self._update_ability_estimates(session)
        
        # Check if complete
        if self._should_stop(session):
            session.is_complete = True
            session.completion_reason = self._get_completion_reason(session)
        
        return session
    
    def _update_ability_estimates(self, session: TestSessionState) -> None:
        """Update ability estimates for all dimensions."""
        # Group responses by dimension
        dimension_responses: Dict[str, List[Tuple[str, bool]]] = {}
        
        for item_id, response in session.responses.items():
            dim = self.selector._get_item_dimension(item_id)
            if dim not in dimension_responses:
                dimension_responses[dim] = []
            dimension_responses[dim].append((item_id, response))
        
        # Calculate theta for each dimension
        for dim, responses in dimension_responses.items():
            # Get IRT parameters
            item_params = []
            response_list = []
            
            for item_id, resp in responses:
                params_result = self.selector.irt_db.get_parameters(item_id)
                if params_result:
                    item_params.append(params_result[0])
                    response_list.append(resp)
            
            if item_params:
                theta_score = self.scoring.calculate_theta(
                    response_list, item_params, dimension=dim
                )
                session.ability_estimates[dim] = theta_score
    
    def _should_stop(self, session: TestSessionState) -> bool:
        """Check if session should stop based on configured rules."""
        config = session.config
        n_items = len(session.administered_items)
        
        for rule in config.stopping_rules:
            if rule == StoppingRule.FIXED_LENGTH:
                if n_items >= config.max_items:
                    return True
                    
            elif rule == StoppingRule.STANDARD_ERROR:
                # Check if all dimensions have SE below target
                all_precise = all(
                    est.standard_error < config.target_se
                    for est in session.ability_estimates.values()
                )
                if all_precise and n_items >= config.min_items:
                    return True
                    
            elif rule == StoppingRule.INFORMATION:
                # Check total information
                total_info = sum(
                    est.information for est in session.ability_estimates.values()
                )
                if total_info > config.target_information and n_items >= config.min_items:
                    return True
                    
            elif rule == StoppingRule.CONFIDENCE_WIDTH:
                # Check confidence interval width
                all_narrow = all(
                    (est.ci_upper - est.ci_lower) < config.max_confidence_width
                    for est in session.ability_estimates.values()
                )
                if all_narrow and n_items >= config.min_items:
                    return True
                    
            elif rule == StoppingRule.TOKEN_BUDGET:
                if session.tokens_used >= config.token_budget:
                    return True
        
        return False
    
    def _get_completion_reason(self, session: TestSessionState) -> str:
        """Get reason for completion."""
        config = session.config
        n_items = len(session.administered_items)
        
        if n_items >= config.max_items:
            return "max_items_reached"
        
        if session.tokens_used >= config.token_budget:
            return "token_budget_exhausted"
        
        # Check which precision target was met
        all_precise = all(
            est.standard_error < config.target_se
            for est in session.ability_estimates.values()
        )
        if all_precise:
            return "target_precision_achieved"
        
        return "unknown"
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get final results for a session."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            "session_id": session_id,
            "is_complete": session.is_complete,
            "completion_reason": session.completion_reason,
            "items_administered": len(session.administered_items),
            "tokens_used": session.tokens_used,
            "ability_estimates": {
                dim: est.to_dict()
                for dim, est in session.ability_estimates.items()
            },
            "dimension_summary": self._summarize_dimensions(session),
        }
    
    def _summarize_dimensions(self, session: TestSessionState) -> Dict[str, Any]:
        """Summarize performance across dimensions."""
        estimates = session.ability_estimates
        
        if not estimates:
            return {}
        
        # Average theta across dimensions
        avg_theta = sum(est.theta for est in estimates.values()) / len(estimates)
        
        # Average reliability
        avg_reliability = sum(est.reliability for est in estimates.values()) / len(estimates)
        
        # Dimension with highest/lowest ability
        sorted_dims = sorted(estimates.items(), key=lambda x: x[1].theta, reverse=True)
        
        return {
            "average_theta": round(avg_theta, 3),
            "average_reliability": round(avg_reliability, 3),
            "strongest_dimension": sorted_dims[0][0] if sorted_dims else None,
            "weakest_dimension": sorted_dims[-1][0] if sorted_dims else None,
            "dimension_count": len(estimates),
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End session and return final results."""
        results = self.get_session_results(session_id)
        
        # Clean up
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        logger.info(f"Ended adaptive test session {session_id}")
        
        return results


def get_adaptive_engine(
    selector: Optional[AdaptiveTestSelector] = None,
    scoring: Optional[ThetaScoringEngine] = None,
) -> AdaptiveTestEngine:
    """Get adaptive testing engine instance."""
    return AdaptiveTestEngine(selector, scoring)
