"""Handler modules for LLM Inspector API."""
from app.handlers.runs import (
    handle_list_runs,
    handle_create_run,
    handle_get_run,
    handle_delete_run,
    handle_cancel_run,
    handle_retry_run,
    handle_continue_run,
    handle_skip_testing,
)
from app.handlers.reports import (
    handle_get_report,
    handle_export_report_csv,
    handle_export_radar_svg,
    handle_export_runs_zip,
    handle_get_responses,
    handle_get_scorecard,
    handle_get_extraction_audit,
    handle_get_theta_report,
    handle_get_pairwise,
)
from app.handlers.baselines import (
    handle_benchmarks,
    handle_create_baseline,
    handle_list_baselines,
    handle_compare_baseline,
    handle_delete_baseline,
    handle_get_baseline,
)
from app.handlers.compare import (
    handle_create_compare_run,
    handle_get_compare_run,
    handle_list_compare_runs,
)
from app.handlers.calibration import (
    handle_calibration_rebuild,
    handle_calibration_snapshot_only,
    handle_create_calibration_replay,
    handle_get_calibration_replay,
    handle_list_calibration_replays,
)
from app.handlers.models import (
    handle_model_trend,
    handle_leaderboard,
    handle_model_theta_trend,
    handle_theta_leaderboard,
)
from app.handlers.misc import (
    handle_health,
    handle_generate_isomorphic,
    handle_static,
)

__all__ = [
    "handle_health",
    "handle_list_runs",
    "handle_create_run",
    "handle_get_run",
    "handle_delete_run",
    "handle_cancel_run",
    "handle_retry_run",
    "handle_continue_run",
    "handle_skip_testing",
    "handle_get_report",
    "handle_export_report_csv",
    "handle_export_radar_svg",
    "handle_export_runs_zip",
    "handle_get_responses",
    "handle_get_scorecard",
    "handle_get_extraction_audit",
    "handle_get_theta_report",
    "handle_get_pairwise",
    "handle_benchmarks",
    "handle_create_baseline",
    "handle_list_baselines",
    "handle_compare_baseline",
    "handle_delete_baseline",
    "handle_get_baseline",
    "handle_create_compare_run",
    "handle_get_compare_run",
    "handle_list_compare_runs",
    "handle_calibration_rebuild",
    "handle_calibration_snapshot_only",
    "handle_create_calibration_replay",
    "handle_get_calibration_replay",
    "handle_list_calibration_replays",
    "handle_model_trend",
    "handle_leaderboard",
    "handle_model_theta_trend",
    "handle_theta_leaderboard",
    "handle_generate_isomorphic",
    "handle_static",
]
