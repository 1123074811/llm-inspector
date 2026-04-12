"""IRT (Item Response Theory) parameter database for LLM Inspector v8.0.

Implements 2PL (Two-Parameter Logistic) model with Fisher information calculation.

Reference: Embretson & Reise (2000) "Item Response Theory for Psychologists"
"""

import math
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..core.provenance import DataProvenance, ProvenanceTracker, get_provenance_tracker


@dataclass
class IRTParameters:
    """IRT 2PL model parameters for a test case.
    
    The 2PL model probability of correct response:
    P(X=1|θ) = c + (1-c) / (1 + exp(-a(θ-b)))
    
    Where:
    - a = discrimination (slope)
    - b = difficulty (location)
    - c = guessing parameter (lower asymptote)
    - θ = latent ability parameter
    
    Reference: Embretson & Reise (2000), Eq. 2.1
    """
    
    case_id: str
    a: float  # Discrimination (ideal range: 0.5-2.0)
    b: float  # Difficulty (ideal range: -3 to 3)
    c: float = 0.25  # Guessing parameter (fixed for 4-option items)
    
    # Quality metrics
    fit_rmse: float = 0.0  # Root mean square error of fit
    info_max: float = 0.0  # Maximum information value
    
    # Validity flags
    is_valid: bool = True
    validation_notes: str = ""
    
    def probability_correct(self, theta: float) -> float:
        """Calculate probability of correct response at ability level θ.
        
        Args:
            theta: Ability parameter (-4 to 4)
            
        Returns:
            Probability of correct response (0-1)
        """
        return self.c + (1 - self.c) / (1 + math.exp(-self.a * (theta - self.b)))
    
    def calculate_information(self, theta: float) -> float:
        """Calculate Fisher information at ability level θ.
        
        Formula: I(θ) = a² * P(θ) * (1-P(θ)) / (1-c)²
        
        Higher information means more precise measurement at this ability level.
        
        Reference: Embretson & Reise (2000), Eq. 5.8
        
        Args:
            theta: Ability parameter (-4 to 4)
            
        Returns:
            Fisher information value
        """
        p = self.probability_correct(theta)
        return (self.a ** 2 * p * (1 - p)) / ((1 - self.c) ** 2)
    
    def get_optimal_ability_range(self) -> Tuple[float, float]:
        """Get the ability range where this item provides good information.
        
        Returns:
            Tuple of (min_theta, max_theta) where information > 0.5 * max_info
        """
        if self.info_max <= 0:
            return (-4.0, 4.0)
        
        threshold = 0.5 * self.info_max
        
        # Find range where information > threshold
        min_theta, max_theta = -4.0, 4.0
        for theta in [x * 0.1 for x in range(-40, 41)]:
            info = self.calculate_information(theta)
            if info >= threshold:
                min_theta = min(min_theta, theta)
                max_theta = max(max_theta, theta)
        
        return (min_theta, max_theta)
    
    def validate(self) -> Dict[str, Any]:
        """Validate IRT parameters according to quality standards.
        
        Returns:
            Validation results with issues found
        """
        issues = []
        
        # Check discrimination
        if self.a < 0.5:
            issues.append(f"Low discrimination (a={self.a:.2f} < 0.5)")
        elif self.a > 2.0:
            issues.append(f"High discrimination (a={self.a:.2f} > 2.0)")
        
        # Check difficulty
        if self.b < -3.0:
            issues.append(f"Very easy item (b={self.b:.2f} < -3.0)")
        elif self.b > 3.0:
            issues.append(f"Very difficult item (b={self.b:.2f} > 3.0)")
        
        # Check fit
        if self.fit_rmse > 0.1:
            issues.append(f"Poor fit (RMSE={self.fit_rmse:.3f} > 0.1)")
        
        # Check info_max
        if self.info_max < 0.1:
            issues.append(f"Low maximum information (info_max={self.info_max:.3f})")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0, 1.0 - len(issues) * 0.2)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "fit_rmse": self.fit_rmse,
            "info_max": self.info_max,
            "is_valid": self.is_valid,
            "validation_notes": self.validation_notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRTParameters':
        """Create from dictionary."""
        return cls(
            case_id=data["case_id"],
            a=data["a"],
            b=data["b"],
            c=data.get("c", 0.25),
            fit_rmse=data.get("fit_rmse", 0.0),
            info_max=data.get("info_max", 0.0),
            is_valid=data.get("is_valid", True),
            validation_notes=data.get("validation_notes", ""),
        )


class IRTParameterDB:
    """IRT parameter database with SQLite backend.
    
    Manages IRT parameters for all test cases with full provenance tracking.
    """
    
    # Parameter quality thresholds
    MIN_DISCRIMINATION = 0.5
    TARGET_DISCRIMINATION = 1.0
    MIN_DIFFICULTY = -3.0
    MAX_DIFFICULTY = 3.0
    MAX_RMSE = 0.1
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            # Default to project irt_data directory
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "irt_data" / "irt_parameters.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # IRT parameters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS irt_parameters (
                    case_id TEXT PRIMARY KEY,
                    a REAL NOT NULL,
                    b REAL NOT NULL,
                    c REAL DEFAULT 0.25,
                    fit_rmse REAL DEFAULT 0.0,
                    info_max REAL DEFAULT 0.0,
                    is_valid INTEGER DEFAULT 1,
                    validation_notes TEXT,
                    calibration_version TEXT,
                    calibrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Provenance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_provenance (
                    case_id TEXT PRIMARY KEY,
                    source_type TEXT,
                    source_id TEXT,
                    sample_size INTEGER,
                    confidence REAL,
                    verified INTEGER,
                    FOREIGN KEY (case_id) REFERENCES irt_parameters(case_id)
                )
            """)
            
            conn.commit()
    
    def store_parameters(
        self, 
        params: IRTParameters, 
        provenance: DataProvenance,
        calibration_version: str = "v2026q1"
    ) -> bool:
        """Store IRT parameters with provenance.
        
        Args:
            params: IRT parameters to store
            provenance: Data provenance information
            calibration_version: Calibration version identifier
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store parameters
                cursor.execute("""
                    INSERT OR REPLACE INTO irt_parameters 
                    (case_id, a, b, c, fit_rmse, info_max, is_valid, 
                     validation_notes, calibration_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    params.case_id,
                    params.a,
                    params.b,
                    params.c,
                    params.fit_rmse,
                    params.info_max,
                    int(params.is_valid),
                    params.validation_notes,
                    calibration_version
                ))
                
                # Store provenance
                cursor.execute("""
                    INSERT OR REPLACE INTO parameter_provenance
                    (case_id, source_type, source_id, sample_size, confidence, verified)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    params.case_id,
                    provenance.source_type,
                    provenance.source_id,
                    provenance.sample_size,
                    provenance.confidence,
                    int(provenance.verified)
                ))
                
                conn.commit()
                
                # Register with global tracker
                tracker = get_provenance_tracker()
                tracker.register(f"irt_params:{params.case_id}", provenance)
                
                return True
                
        except sqlite3.Error as e:
            print(f"Error storing IRT parameters for {params.case_id}: {e}")
            return False
    
    def get_parameters(self, case_id: str) -> Optional[Tuple[IRTParameters, DataProvenance]]:
        """Get IRT parameters and provenance for a case.
        
        Args:
            case_id: Test case identifier
            
        Returns:
            Tuple of (IRTParameters, DataProvenance) or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get parameters
                cursor.execute("""
                    SELECT * FROM irt_parameters WHERE case_id = ?
                """, (case_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                params = IRTParameters(
                    case_id=row["case_id"],
                    a=row["a"],
                    b=row["b"],
                    c=row["c"],
                    fit_rmse=row["fit_rmse"],
                    info_max=row["info_max"],
                    is_valid=bool(row["is_valid"]),
                    validation_notes=row["validation_notes"] or "",
                )
                
                # Get provenance
                cursor.execute("""
                    SELECT * FROM parameter_provenance WHERE case_id = ?
                """, (case_id,))
                
                prov_row = cursor.fetchone()
                if prov_row:
                    provenance = DataProvenance(
                        source_type=prov_row["source_type"],
                        source_id=prov_row["source_id"],
                        collected_at=row["calibrated_at"],
                        sample_size=prov_row["sample_size"],
                        confidence=prov_row["confidence"],
                        verified=bool(prov_row["verified"]),
                        verified_by="irt_calibration_system",
                    )
                else:
                    provenance = DataProvenance.create_fallback(
                        f"irt_params_{case_id}",
                        "No provenance found in database"
                    )
                
                return (params, provenance)
                
        except sqlite3.Error as e:
            print(f"Error retrieving IRT parameters for {case_id}: {e}")
            return None
    
    def get_all_parameters(self) -> Dict[str, Tuple[IRTParameters, DataProvenance]]:
        """Get all IRT parameters with provenance.
        
        Returns:
            Dictionary mapping case_id to (IRTParameters, DataProvenance)
        """
        results = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT case_id FROM irt_parameters")
                rows = cursor.fetchall()
                
                for (case_id,) in rows:
                    result = self.get_parameters(case_id)
                    if result:
                        results[case_id] = result
                        
        except sqlite3.Error as e:
            print(f"Error retrieving all IRT parameters: {e}")
        
        return results
    
    def get_parameters_by_dimension(self, dimension: str) -> List[IRTParameters]:
        """Get all parameters for a specific dimension.
        
        Args:
            dimension: Dimension prefix (e.g., "reasoning", "coding")
            
        Returns:
            List of IRTParameters
        """
        params_list = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM irt_parameters 
                    WHERE case_id LIKE ?
                """, (f"{dimension}%",))
                
                for row in cursor.fetchall():
                    params = IRTParameters(
                        case_id=row["case_id"],
                        a=row["a"],
                        b=row["b"],
                        c=row["c"],
                        fit_rmse=row["fit_rmse"],
                        info_max=row["info_max"],
                        is_valid=bool(row["is_valid"]),
                        validation_notes=row["validation_notes"] or "",
                    )
                    params_list.append(params)
                    
        except sqlite3.Error as e:
            print(f"Error retrieving parameters by dimension: {e}")
        
        return params_list
    
    def calculate_dimension_information(
        self, 
        dimension: str, 
        theta: float
    ) -> Dict[str, Any]:
        """Calculate total information for a dimension at ability level θ.
        
        Args:
            dimension: Dimension name
            theta: Ability level
            
        Returns:
            Information statistics
        """
        params_list = self.get_parameters_by_dimension(dimension)
        
        if not params_list:
            return {
                "dimension": dimension,
                "total_information": 0.0,
                "item_count": 0,
                "average_information": 0.0,
                "max_information": 0.0,
            }
        
        information_values = [
            p.calculate_information(theta) for p in params_list if p.is_valid
        ]
        
        if not information_values:
            return {
                "dimension": dimension,
                "total_information": 0.0,
                "item_count": 0,
                "average_information": 0.0,
                "max_information": 0.0,
            }
        
        total_info = sum(information_values)
        
        return {
            "dimension": dimension,
            "total_information": total_info,
            "standard_error": 1.0 / math.sqrt(total_info) if total_info > 0 else float('inf'),
            "item_count": len(information_values),
            "average_information": total_info / len(information_values),
            "max_information": max(information_values),
            "reliability": total_info / (total_info + 1) if total_info > 0 else 0.0,
        }
    
    def select_optimal_items(
        self,
        dimension: str,
        theta: float,
        n_items: int = 5,
        exclude_case_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Select items with maximum information at ability level θ.
        
        Args:
            dimension: Dimension to select from
            theta: Current ability estimate
            n_items: Number of items to select
            exclude_case_ids: Case IDs to exclude (already used)
            
        Returns:
            List of (case_id, information) tuples, sorted by information
        """
        exclude_set = set(exclude_case_ids or [])
        
        params_list = [
            p for p in self.get_parameters_by_dimension(dimension)
            if p.is_valid and p.case_id not in exclude_set
        ]
        
        # Calculate information for each item
        item_info = [
            (p.case_id, p.calculate_information(theta))
            for p in params_list
        ]
        
        # Sort by information (descending)
        item_info.sort(key=lambda x: x[1], reverse=True)
        
        return item_info[:n_items]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total count
                cursor.execute("SELECT COUNT(*) FROM irt_parameters")
                total_count = cursor.fetchone()[0]
                
                # Valid count
                cursor.execute("SELECT COUNT(*) FROM irt_parameters WHERE is_valid = 1")
                valid_count = cursor.fetchone()[0]
                
                # Average parameters
                cursor.execute("""
                    SELECT AVG(a), AVG(b), AVG(fit_rmse), AVG(info_max)
                    FROM irt_parameters WHERE is_valid = 1
                """)
                avg_a, avg_b, avg_rmse, avg_info = cursor.fetchone()
                
                # Parameter distribution
                cursor.execute("""
                    SELECT 
                        SUM(CASE WHEN a >= 1.0 THEN 1 ELSE 0 END) as good_discrimination,
                        SUM(CASE WHEN b BETWEEN -2 AND 2 THEN 1 ELSE 0 END) as normal_difficulty
                    FROM irt_parameters WHERE is_valid = 1
                """)
                good_disc, normal_diff = cursor.fetchone()
                
                return {
                    "total_parameters": total_count,
                    "valid_parameters": valid_count,
                    "validity_rate": valid_count / total_count if total_count > 0 else 0.0,
                    "average_discrimination": avg_a or 0.0,
                    "average_difficulty": avg_b or 0.0,
                    "average_rmse": avg_rmse or 0.0,
                    "average_info_max": avg_info or 0.0,
                    "good_discrimination_count": good_disc or 0,
                    "normal_difficulty_count": normal_diff or 0,
                }
                
        except sqlite3.Error as e:
            print(f"Error getting statistics: {e}")
            return {"total_parameters": 0}
    
    def export_to_json(self, output_path: str) -> bool:
        """Export all parameters to JSON file.
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            True if successful
        """
        try:
            all_params = self.get_all_parameters()
            
            export_data = {
                "export_timestamp": "2026-04-11T00:00:00Z",
                "version": "v2026q1",
                "parameters": {
                    case_id: {
                        "params": params.to_dict(),
                        "provenance": provenance.to_dict()
                    }
                    for case_id, (params, provenance) in all_params.items()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting parameters: {e}")
            return False
    
    def import_from_json(self, input_path: str, calibration_version: str = "v2026q1") -> int:
        """Import parameters from JSON file.
        
        Args:
            input_path: Path to input JSON file
            calibration_version: Version to assign to imported parameters
            
        Returns:
            Number of parameters imported
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            
            for case_id, case_data in data.get("parameters", {}).items():
                params_dict = case_data.get("params", {})
                prov_dict = case_data.get("provenance", {})
                
                params = IRTParameters.from_dict(params_dict)
                
                provenance = DataProvenance(
                    source_type=prov_dict.get("source_type", "irt_calibration"),
                    source_id=prov_dict.get("source_id", f"imported_{case_id}"),
                    collected_at=prov_dict.get("collected_at", "2026-04-11T00:00:00Z"),
                    sample_size=prov_dict.get("sample_size", 0),
                    confidence=prov_dict.get("confidence", 0.5),
                    verified=prov_dict.get("verified", False),
                    verified_by=prov_dict.get("verified_by"),
                )
                
                if self.store_parameters(params, provenance, calibration_version):
                    imported_count += 1
            
            return imported_count
            
        except Exception as e:
            print(f"Error importing parameters: {e}")
            return 0


class ThetaScoreConverter:
    """Convert between percent scores and theta ability estimates.
    
    Reference: Embretson & Reise (2000), Maximum Likelihood Estimation
    """
    
    @staticmethod
    def percent_to_theta(
        percentile: float, 
        item_params: List[IRTParameters],
        max_iterations: int = 50,
        convergence_threshold: float = 0.001
    ) -> Tuple[float, float]:
        """Convert percent score to theta ability estimate.
        
        Uses Newton-Raphson iteration for maximum likelihood estimation.
        
        Args:
            percentile: Raw percent score (0-100)
            item_params: List of IRT parameters for items in the dimension
            max_iterations: Maximum number of iterations
            convergence_threshold: Convergence threshold for delta
            
        Returns:
            Tuple of (theta_estimate, standard_error)
        """
        if not item_params:
            return (0.0, 999.0)  # Undefined
        
        # Initial theta estimate
        theta = 0.0
        
        for iteration in range(max_iterations):
            # Calculate Fisher information
            fisher_info = sum(
                p.calculate_information(theta) for p in item_params
            )
            
            # Calculate score residual
            score_residual = 0.0
            for p in item_params:
                prob = p.probability_correct(theta)
                if prob > 0 and prob < 1:
                    score_residual += (percentile / 100 - prob) / (prob * (1 - prob))
            
            # Newton-Raphson update
            if fisher_info > 0:
                delta = score_residual / fisher_info
                theta += delta
                
                if abs(delta) < convergence_threshold:
                    break
            else:
                break
        
        # Final standard error
        final_info = sum(p.calculate_information(theta) for p in item_params)
        standard_error = 1.0 / math.sqrt(final_info) if final_info > 0 else 999.0
        
        # Clamp to valid range
        theta = max(-4.0, min(4.0, theta))
        
        return (theta, standard_error)
    
    @staticmethod
    def theta_to_percentile(theta: float) -> float:
        """Convert theta to cumulative normal distribution percentile.
        
        Args:
            theta: Ability estimate (-4 to 4)
            
        Returns:
            Percentile score (0-100)
        """
        try:
            from scipy.stats import norm
            return norm.cdf(theta) * 100
        except ImportError:
            # Fallback: approximate normal CDF
            # Using error function approximation
            import math
            # Phi(x) ≈ 0.5 * [1 + erf(x / sqrt(2))]
            return 100 * (0.5 * (1 + math.erf(theta / math.sqrt(2))))


def get_calibrated_params(case_id: str) -> Optional[IRTParameters]:
    """Get calibrated parameters for a test case from the database.
    
    Args:
        case_id: Test case identifier
        
    Returns:
        IRTParameters or None if not found
    """
    db = get_irt_db()
    result = db.get_parameters(case_id)
    if result:
        params, _ = result
        return params
    return None


def get_irt_db(db_path: Optional[str] = None) -> IRTParameterDB:
    """Get IRT parameter database instance."""
    return IRTParameterDB(db_path)
