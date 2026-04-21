def get_evaluation_mode(custom_jd):
    """Strict mode switch: JD present => JD_MODE, otherwise ROLE_MODE."""
    return "JD_MODE" if custom_jd else "ROLE_MODE"

