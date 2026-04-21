import io
import re
from pypdf import PdfReader

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "will", "your", "you",
    "our", "are", "not", "but", "all", "any", "can", "should", "must", "such", "using",
}
FILLER_WORDS = {
    "with", "and", "on", "in", "of", "to", "for", "a", "an", "the", "from", "by",
    "work", "working", "hands", "hand", "using", "across", "through", "ability",
}

TARGET_SECTIONS = (
    "job responsibilities",
    "responsibilities",
    "skills",
    "requirements",
    "qualifications",
)

IGNORE_SECTION_HINTS = (
    "about company",
    "about us",
    "who we are",
    "mission",
    "vision",
    "awards",
    "recognition",
    "achievements",
    "benefits",
)

INVALID_TOKENS = {
    "award", "awards", "globally", "employees", "company", "vision", "mission",
    "culture", "values", "office", "brand", "industry", "clients",
}


def _extract_text_from_jd_file(jd_file):
    ext = jd_file.name.split(".")[-1].lower()
    if ext == "pdf":
        data = jd_file.getvalue() if hasattr(jd_file, "getvalue") else jd_file.read()
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == "txt":
        data = jd_file.getvalue() if hasattr(jd_file, "getvalue") else jd_file.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)
    return ""


def _extract_candidate_terms(text):
    # Capture tech-like tokens strictly from JD text (no global injection).
    raw = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-/]{1,30}", text)
    terms = []
    for token in raw:
        t = token.strip().lower()
        if t in STOPWORDS or len(t) < 3:
            continue
        terms.append(t)
    return terms


def _split_jd_sections(jd_text):
    """
    Parse JD into named sections by heading-like lines.
    Keeps extraction focused on requirement-relevant blocks only.
    """
    lines = [ln.strip() for ln in jd_text.splitlines()]
    sections = {}
    current = "misc"
    sections[current] = []
    for line in lines:
        if not line:
            continue
        normalized = line.lower().strip(":").strip()
        is_heading = (
            len(line) <= 60
            and any(hint in normalized for hint in TARGET_SECTIONS + IGNORE_SECTION_HINTS)
        )
        if is_heading:
            current = normalized
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _is_target_section(section_name):
    name = (section_name or "").lower()
    if any(h in name for h in IGNORE_SECTION_HINTS):
        return False
    return any(h in name for h in TARGET_SECTIONS)


def _extract_phrases(text):
    """
    Phrase-based extraction: captures meaningful requirement chunks
    instead of noisy single-word tokens.
    """
    chunks = re.split(r"[,\n;•\-\u2022]+", text)
    phrases = []
    for chunk in chunks:
        c = " ".join(chunk.strip().split())
        if len(c) < 4:
            continue
        lc = c.lower()
        if any(t in lc for t in INVALID_TOKENS):
            continue
        # keep phrases up to 5 words to preserve requirement semantics
        words = re.findall(r"[a-zA-Z0-9\+\#\./]+", lc)
        if 1 <= len(words) <= 5:
            phrase = clean_phrase(" ".join(words))
            if phrase not in STOPWORDS:
                phrases.append(phrase)
    return phrases


def clean_phrase(text):
    words = re.findall(r"[a-zA-Z0-9\+\#\./]+", text.lower())
    cleaned = [w for w in words if w not in STOPWORDS and w not in FILLER_WORDS and w not in INVALID_TOKENS]
    if not cleaned:
        return ""

    joined = " ".join(cleaned[:4])
    # normalize common concepts
    if any(t in joined for t in ("openai", "llm", "gpt", "prompt")):
        return "ai tools"
    if any(t in joined for t in ("api", "integration", "integrations", "endpoint", "backend")):
        return "apis integrations"
    if any(t in joined for t in ("automation", "workflow", "agent")):
        return "automation workflows"
    return joined.strip()


def is_valid_skill(term):
    t = term.lower().strip()
    if not t or t in INVALID_TOKENS:
        return False
    # technical / requirement-oriented intent
    technical_signals = (
        "api", "ai", "ml", "python", "java", "javascript", "typescript", "react", "node",
        "prompt", "automation", "workflow", "integration", "sql", "cloud", "docker",
        "kubernetes", "llm", "model", "data", "analytics", "testing", "microservice",
    )
    requirement_signals = ("experience", "knowledge", "proficiency", "ability", "hands-on")
    return any(sig in t for sig in technical_signals) or any(sig in t for sig in requirement_signals)


def _extract_requirement_lines(lines):
    markers = ("require", "must", "need", "responsibil", "experience", "proficien", "familiar")
    return [line for line in lines if any(m in line.lower() for m in markers)]


def extract_jd_requirements(jd_text):
    lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    sections = _split_jd_sections(jd_text)
    target_lines = []
    for sec_name, sec_lines in sections.items():
        if _is_target_section(sec_name):
            target_lines.extend(sec_lines)
    if not target_lines:
        # fallback to requirement-like lines only, still avoid company-marketing blocks
        target_lines = _extract_requirement_lines(lines)

    base_text = "\n".join(target_lines)
    phrases = _extract_phrases(base_text)
    valid_phrases = [p for p in phrases if is_valid_skill(p)]

    # Deduplicate while preserving order
    seen = set()
    keywords = []
    for phrase in valid_phrases:
        if phrase not in seen:
            seen.add(phrase)
            keywords.append(phrase)

    tools = [k for k in keywords if any(x in k for x in ("api", "python", "java", "javascript", "typescript", "react", "node", "docker", "kubernetes", "sql", "openai", "llm"))][:20]
    skills = [k for k in keywords if k not in tools][:20]
    concepts = [k for k in keywords if k not in set(skills + tools)][:20]

    responsibilities = target_lines[:20]
    return {
        "skills": skills,
        "tools": tools,
        "concepts": concepts,
        "keywords": keywords[:40],
        "responsibilities": responsibilities,
    }


def parse_jd(jd_file):
    jd_text = _extract_text_from_jd_file(jd_file)
    extracted = extract_jd_requirements(jd_text)
    return {
        "jd_text": jd_text,
        "skills": extracted["skills"],
        "tools": extracted["tools"],
        "concepts": extracted["concepts"],
        "responsibilities": extracted["responsibilities"],
        "keywords": extracted["keywords"],
    }


def build_dynamic_categories(jd_data):
    text = jd_data.get("jd_text", "").lower()

    categories = {}
    if "ai" in text or "ml" in text or "machine learning" in text:
        categories["AI Understanding"] = 25
    if "api" in text or "integration" in text:
        categories["APIs & Integration"] = 25
    if "automation" in text or "workflow" in text:
        categories["Automation / Workflows"] = 20
    if "python" in text or "java" in text or "javascript" in text or "typescript" in text:
        categories["Programming"] = 15
    categories["Projects Relevance"] = 10
    categories["Bonus"] = 5

    # normalize to 100 if needed
    total = sum(categories.values()) or 100
    if total != 100:
        categories = {k: (v * 100.0 / total) for k, v in categories.items()}
    return categories


def is_jd_relevant(skill, jd_text):
    return skill.lower() in (jd_text or "").lower()


def _keyword_overlap_score(resume_text, jd_keywords):
    if not jd_keywords:
        return 0, []
    r = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw in r]
    return int(round((len(matched) / len(jd_keywords)) * 100)), matched


def contextual_match(resume_text, jd_keywords):
    rt = (resume_text or "").lower()
    contextual_map = {
        "ai": ["openai", "llm", "gpt", "langchain", "prompt"],
        "ai tools": ["openai", "llm", "gpt", "langchain", "prompt"],
        "ai ml exposure": ["machine learning", "ml", "model", "llm", "nlp"],
        "apis": ["api", "rest", "integration", "endpoint"],
        "apis integrations": ["api", "rest", "integration", "endpoint", "backend"],
        "system integrations": ["api", "integration", "backend", "microservice"],
        "automation": ["workflow", "automate", "pipeline", "agent"],
        "automation workflows": ["workflow", "automate", "pipeline", "agent"],
        "machine": ["ml", "model", "training"],
        "javascript": ["node", "react", "typescript"],
        "python": ["pandas", "numpy", "flask", "fastapi"],
    }
    matched = []
    for kw in jd_keywords:
        if kw in rt:
            continue
        for anchor, related in contextual_map.items():
            if anchor in kw and any(r in rt for r in related):
                matched.append(kw)
                break
    if not jd_keywords:
        return 0, []
    score = int(round((len(set(matched)) / len(jd_keywords)) * 100))
    return score, sorted(set(matched))


def _project_relevance_score(resume_text):
    rt = resume_text.lower()
    signals = ["project", "implemented", "built", "developed", "deployed", "intern", "experience"]
    hits = sum(1 for s in signals if s in rt)
    return min(100, hits * 12)


def _programming_alignment_score(resume_text, jd_data):
    rt = (resume_text or "").lower()
    candidates = [k for k in jd_data.get("keywords", []) if k in {"python", "java", "javascript", "typescript", "sql", "go", "c++", "c#"}]
    if not candidates:
        return 60  # neutral if JD has no explicit language requirement
    matched = sum(1 for lang in candidates if lang in rt)
    return int(round((matched / len(candidates)) * 100))


def _soft_requirement_factor(jd_text):
    jt = (jd_text or "").lower()
    soft_markers = ("basic", "exposure", "willingness", "nice to have", "preferred")
    return 0.6 if any(m in jt for m in soft_markers) else 1.0


def _trainability_bonus(trainability_score):
    if trainability_score > 70:
        return 6, "strong"
    if trainability_score >= 60:
        return 3, "moderate"
    return 0, "developing"


def _decision_from_jd_score(score):
    if score >= 75:
        return "Selected"
    if score >= 60:
        return "Shortlisted"
    if score >= 50:
        return "Borderline"
    return "Rejected"


def evaluate_jd_mode(agent, resume_text, jd_data, experience_level="mid"):
    categories = build_dynamic_categories(jd_data)
    jd_text = jd_data.get("jd_text", "")
    jd_keywords_all = jd_data.get("keywords", [])
    jd_keywords = [k for k in jd_keywords_all if is_jd_relevant(k, jd_text)]  # anti-contamination gate
    jd_skills_all = jd_data.get("skills", []) + jd_data.get("tools", []) + jd_data.get("concepts", [])
    jd_skills = [s for s in jd_skills_all if is_jd_relevant(s, jd_text)]

    match_score, matched_keywords = _keyword_overlap_score(resume_text, jd_keywords)
    contextual_score, contextual_keywords = contextual_match(resume_text, jd_keywords)
    project_score = _project_relevance_score(resume_text)
    programming_score = _programming_alignment_score(resume_text, jd_data)
    soft_factor = _soft_requirement_factor(jd_text)

    matched_skills = [s for s in jd_skills if s in (resume_text or "").lower()]
    missing_skills = [s for s in jd_skills if s not in (resume_text or "").lower()]
    trainability_score = int(round((match_score + project_score + programming_score) / 3))
    trainability_bonus, trainability_band = _trainability_bonus(trainability_score)

    # Reward-only JD scoring: no negative penalties for irrelevant skills.
    skill_component = match_score * 0.45
    contextual_component = contextual_score * 0.20
    project_component = project_score * 0.20
    programming_component = programming_score * 0.10
    bonus_component = min(5, max(0, len(matched_keywords) - 2)) + trainability_bonus
    # soften strictness where JD signals "basic/exposure/willingness"
    soft_penalty_buffer = (1 - soft_factor) * 8.0
    final_score = int(round(min(100, skill_component + contextual_component + project_component + programming_component + bonus_component + soft_penalty_buffer)))

    strong_core_alignment = (match_score >= 55 or contextual_score >= 60) and project_score >= 45
    if strong_core_alignment:
        final_score = max(final_score, 55)

    strengths = [
        f"Strong alignment with JD keywords: {', '.join(matched_keywords[:5])}" if matched_keywords else "",
        "Contextual evidence of AI/API exposure (e.g., OpenAI/LLM integrations) aligns with JD intent." if contextual_keywords else "",
        "Project experience indicates practical implementation ability relevant to JD responsibilities." if project_score >= 50 else "",
        f"Programming alignment is {programming_score}/100 against JD language requirements.",
    ]
    strengths = [s for s in strengths if s][:4]
    if not strengths:
        strengths = ["Shows partial JD-relevant exposure with room to grow in required areas."]

    gaps = [
        f"Missing direct evidence for: {', '.join(missing_skills[:5])}" if missing_skills else "",
        "Some required JD concepts are not explicitly demonstrated in resume projects." if len(missing_skills) >= 2 else "",
    ]
    gaps = [g for g in gaps if g]
    decision = _decision_from_jd_score(final_score)

    prompt = f"""
You are evaluating a candidate STRICTLY based on the given Job Description.
IGNORE any generic role assumptions.

Rules:
- DO NOT introduce skills not present in JD.
- DO NOT penalize for missing irrelevant tools.
- Focus only on JD alignment.
- Consider contextual matches (e.g., OpenAI API = AI tools).
- Be fair for fresher candidates.

Experience level: {experience_level}

Use this data only:
- JD keywords: {jd_keywords}
- Matched skills: {matched_skills}
- Missing skills: {missing_skills}
- JD match score: {match_score}
- Contextual match score: {contextual_score}
- Programming alignment score: {programming_score}
- Final computed score: {final_score}
- Trainability score: {trainability_score} ({trainability_band})
- Soft requirement factor: {soft_factor}

Write high-quality recruiter output with:
1) JD-aligned strengths
2) JD-aligned gaps
3) Balanced summary
"""
    llm_reason = agent.get_llm_response(prompt)
    if not llm_reason:
        llm_reason = (
            "Candidate evaluation is based strictly on JD alignment. "
            "Profile shows reasonable fit in matched requirements with clear upskilling areas."
        )

    return {
        "mode": "JD_MODE",
        "overall_score": final_score,
        "final_score": final_score,
        "jd_match_score": match_score,
        "decision_band": decision.lower().replace(" ", "_"),
        "final_decision": decision,
        "selected": decision in {"Selected", "Shortlisted"},
        "section_breakdown": {
            "jd_direct_match": {"score": round(match_score / 10, 2), "weight": 45, "weighted_contribution": round(skill_component, 2)},
            "jd_contextual_match": {"score": round(contextual_score / 10, 2), "weight": 20, "weighted_contribution": round(contextual_component, 2)},
            "project_relevance": {"score": round(project_score / 10, 2), "weight": 20, "weighted_contribution": round(project_component, 2)},
            "programming_alignment": {"score": round(programming_score / 10, 2), "weight": 10, "weighted_contribution": round(programming_component, 2)},
            "bonus_trainability": {"score": round((bonus_component * 10) / 10, 2), "weight": 5, "weighted_contribution": round(bonus_component, 2), "categories": categories},
        },
        "strengths": strengths,
        "missing_skills": missing_skills[:10],
        "trainability_score": trainability_score,
        "reasoning": llm_reason,
        "jd_data": jd_data,
        "qualitative_feedback": {
            "verdict": decision,
            "confidence": "Medium",
            "summary": llm_reason,
            "recruiter_comments": strengths[:3],
            "top_improvements": (gaps[:3] if gaps else missing_skills[:3]),
        },
    }

