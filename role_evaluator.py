def evaluate_role_mode(agent, role_requirements):
    """
    Role-based evaluator wrapper.
    Keeps all existing role/experience logic in agent.semantic_skill_analysis.
    """
    return agent.semantic_skill_analysis(role_requirements or [])

