import re
from pypdf import PdfReader
import io
from google.generativeai import GenerativeModel
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import google.generativeai as genai
import time
from mode_router import get_evaluation_mode
from role_evaluator import evaluate_role_mode
from jd_evaluator import parse_jd, evaluate_jd_mode

class ResumeAnalysisAgent:
    ROLE_PROFILES = {
        "AI/ML Engineer": {
            "sections": {
                "core_ml_nlp": {"weight": 0.40, "skills": ["Python", "Machine Learning", "NLP", "Deep Learning"]},
                "mlops_production": {"weight": 0.25, "skills": ["MLOps", "Model Evaluation", "Feature Engineering", "CI/CD"]},
                "modern_ai_stack": {"weight": 0.20, "skills": ["PyTorch", "TensorFlow", "Hugging Face", "LLMs", "LangChain", "RAG"]},
                "projects": {"weight": 0.10, "skills": ["Projects", "Production Deployment", "Work Experience"]},
                "bonus": {"weight": 0.05, "skills": ["Reinforcement Learning", "Computer Vision", "AutoML"]},
            },
            "trainability_sections": ["core_ml_nlp", "projects", "mlops_production"],
            "consistency_rule": "ml_framework_consistency",
        },
        "Data Scientist": {
            "sections": {
                "core_ml_nlp": {"weight": 0.40, "skills": ["Python", "Machine Learning", "NLP", "Statistics"]},
                "mlops_production": {"weight": 0.25, "skills": ["Model Evaluation", "Feature Engineering", "MLOps", "Experimental Design"]},
                "modern_ai_stack": {"weight": 0.20, "skills": ["PyTorch", "TensorFlow", "Hugging Face", "LLMs"]},
                "projects": {"weight": 0.10, "skills": ["Projects", "Work Experience", "A/B Testing"]},
                "bonus": {"weight": 0.05, "skills": ["Computer Vision", "AutoML", "Reinforcement Learning"]},
            },
            "trainability_sections": ["core_ml_nlp", "projects", "mlops_production"],
            "consistency_rule": "ml_framework_consistency",
        },
        "Frontend Engineer": {
            "sections": {
                "core_frontend": {"weight": 0.40, "skills": ["HTML5", "CSS3", "JavaScript", "React"]},
                "frameworks_ui": {"weight": 0.20, "skills": ["Next.js", "TypeScript", "Tailwind CSS", "Redux"]},
                "projects_portfolio": {"weight": 0.20, "skills": ["Projects", "UI Portfolio", "Work Experience"]},
                "tools_apis": {"weight": 0.10, "skills": ["Git", "REST APIs", "GraphQL"]},
                "bonus": {"weight": 0.10, "skills": ["Performance Optimization", "Testing", "Animations"]},
            },
            "trainability_sections": ["core_frontend", "projects_portfolio", "tools_apis"],
        },
        "Backend Engineer": {
            "sections": {
                "core_backend": {"weight": 0.40, "skills": ["Python", "Java", "Node.js", "REST APIs"]},
                "databases": {"weight": 0.20, "skills": ["SQL", "NoSQL", "Redis"]},
                "system_design": {"weight": 0.20, "skills": ["Microservices", "Scalability", "Architecture"]},
                "projects": {"weight": 0.10, "skills": ["Projects", "Work Experience"]},
                "bonus": {"weight": 0.10, "skills": ["Caching", "Message Queues", "RabbitMQ"]},
            },
            "trainability_sections": ["core_backend", "projects", "databases"],
        },
        "Full Stack Developer": {
            "sections": {
                "frontend": {"weight": 0.25, "skills": ["React", "JavaScript", "HTML5", "CSS3"]},
                "backend": {"weight": 0.25, "skills": ["Node.js", "Python", "REST APIs", "Express"]},
                "integration_apis": {"weight": 0.20, "skills": ["Integration", "APIs", "Authentication"]},
                "projects": {"weight": 0.20, "skills": ["Projects", "Work Experience"]},
                "bonus": {"weight": 0.10, "skills": ["CI/CD", "Cloud Services", "Performance Optimization"]},
            },
            "trainability_sections": ["frontend", "projects", "integration_apis"],
        },
        "Data Engineer": {
            "sections": {
                "pipelines_etl": {"weight": 0.35, "skills": ["ETL Pipelines", "Data Pipelines", "Airflow", "DBT"]},
                "databases_warehousing": {"weight": 0.25, "skills": ["SQL", "Data Warehousing", "Snowflake", "BigQuery"]},
                "big_data_tools": {"weight": 0.20, "skills": ["Spark", "Hadoop", "Kafka"]},
                "projects": {"weight": 0.10, "skills": ["Projects", "Work Experience"]},
                "bonus": {"weight": 0.10, "skills": ["AWS Glue", "Redshift", "Streaming"]},
            },
            "trainability_sections": ["pipelines_etl", "projects", "databases_warehousing"],
        },
        "DevOps Engineer": {
            "sections": {
                "cicd": {"weight": 0.30, "skills": ["CI/CD", "Jenkins", "GitHub Actions"]},
                "cloud": {"weight": 0.25, "skills": ["AWS", "GCP", "Azure"]},
                "containers": {"weight": 0.20, "skills": ["Docker", "Kubernetes", "Helm"]},
                "monitoring_infra": {"weight": 0.15, "skills": ["Prometheus", "Grafana", "Terraform"]},
                "bonus": {"weight": 0.10, "skills": ["SRE", "Security", "Networking"]},
            },
            "trainability_sections": ["cicd", "monitoring_infra", "cloud"],
        },
        "Product Manager": {
            "sections": {
                "product_thinking": {"weight": 0.30, "skills": ["Product Strategy", "Roadmapping", "Prioritization"]},
                "case_studies_projects": {"weight": 0.25, "skills": ["Projects", "Product Lifecycle", "User Stories"]},
                "analytics_metrics": {"weight": 0.20, "skills": ["Data Analysis", "KPI Definition", "A/B Testing"]},
                "communication": {"weight": 0.15, "skills": ["Stakeholder Management", "User Research", "Communication"]},
                "bonus": {"weight": 0.10, "skills": ["Competitive Analysis", "Customer Journey Mapping", "Agile Methodologies"]},
            },
            "trainability_sections": ["product_thinking", "case_studies_projects", "analytics_metrics"],
        },
    }

    # Tunable moderation policy for modern AI stack penalties.
    MODERN_AI_CORE_STRONG_THRESHOLD = 7.0
    MODERN_AI_PRODUCTION_STRONG_THRESHOLD = 6.0
    MODERN_AI_CORE_VERY_STRONG_THRESHOLD = 7.5
    MODERN_AI_FLOOR_STRONG = 5.5
    MODERN_AI_FLOOR_VERY_STRONG = 5.0

    DEFAULT_ROLE = "AI/ML Engineer"
    MAX_TOTAL_PENALTY = 12

    def __init__(self, api_key, cutoff_score=70):
        self.api_key = api_key
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.vectorstore = None
        self.analysis_result = None
        self.current_role = self.DEFAULT_ROLE

        genai.configure(api_key=api_key)

        self.model_names = self._discover_model_names()
        self._models = {}
        self.disabled_models = set()
        self.preferred_model = None

    def _decision_band(self, score):
        if score >= 80:
            return "strong_hire"
        if score >= 65:
            return "shortlist"
        if score >= 55:
            return "borderline"
        return "reject"

    def _decision_label(self, decision_band):
        mapping = {
            "strong_hire": "Selected",
            "shortlist": "Shortlisted",
            "borderline": "Borderline",
            "reject": "Rejected",
        }
        return mapping.get(decision_band, "Borderline")

    def _extract_experience_years(self):
        text = (self.resume_text or "").lower()
        patterns = [
            r"(\d+)\+?\s+years",
            r"(\d+)\+?\s+yrs",
            r"experience\s+of\s+(\d+)",
        ]
        candidates = []
        for pattern in patterns:
            for match in re.findall(pattern, text):
                try:
                    candidates.append(int(match))
                except ValueError:
                    continue
        return max(candidates) if candidates else 0

    def _experience_level(self):
        years = self._extract_experience_years()
        if years < 2:
            return "fresher", years
        if years < 5:
            return "mid", years
        return "senior", years

    def _adjust_role_weights_for_experience(self, role_profile, level):
        # reward fundamentals/projects for freshers; emphasize production stack for seniors
        adjusted_sections = {}
        for name, meta in role_profile["sections"].items():
            adjusted_sections[name] = {"weight": meta["weight"], "skills": list(meta["skills"])}

        def shift(from_keys, to_keys, delta):
            available = sum(adjusted_sections[k]["weight"] for k in from_keys if k in adjusted_sections)
            if available <= 0:
                return
            delta_effective = min(delta, available)
            per_from = delta_effective / max(1, len(from_keys))
            per_to = delta_effective / max(1, len(to_keys))
            for k in from_keys:
                if k in adjusted_sections:
                    adjusted_sections[k]["weight"] = max(0.05, adjusted_sections[k]["weight"] - per_from)
            for k in to_keys:
                if k in adjusted_sections:
                    adjusted_sections[k]["weight"] += per_to

        if level == "fresher":
            shift(
                from_keys=[k for k in adjusted_sections if ("framework" in k or "tools" in k or "modern" in k)],
                to_keys=[k for k in adjusted_sections if ("core" in k or "project" in k or "case" in k)],
                delta=0.12,
            )
        elif level == "senior":
            shift(
                from_keys=[k for k in adjusted_sections if ("project" in k or "bonus" in k)],
                to_keys=[k for k in adjusted_sections if ("framework" in k or "tools" in k or "modern" in k or "system" in k or "mlops" in k)],
                delta=0.08,
            )

        # normalize to 1.0
        total = sum(m["weight"] for m in adjusted_sections.values()) or 1.0
        for m in adjusted_sections.values():
            m["weight"] = m["weight"] / total
        return {"sections": adjusted_sections, "trainability_sections": role_profile.get("trainability_sections", []), "consistency_rule": role_profile.get("consistency_rule")}

    def _discover_model_names(self):
        """Discover models that support generateContent for this API key."""
        preferred_order = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
            "models/gemini-2.0-flash",
            "models/gemini-pro",
            "models/gemini-1.0-pro",
        ]
        try:
            available = []
            for m in genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    available.append(m.name)

            if not available:
                print("No generateContent models returned by API.")
                return preferred_order

            # Keep preferred models first (if available), then include any other valid Gemini models.
            ordered = [name for name in preferred_order if name in available]
            for name in available:
                if name.startswith("models/gemini") and name not in ordered:
                    ordered.append(name)

            print("Discovered Gemini models:", ordered)
            return ordered or preferred_order
        except Exception as e:
            print(f"Model discovery failed: {e}")
            return preferred_order

    def _get_or_create_model(self, model_name):
        if model_name not in self._models:
            self._models[model_name] = GenerativeModel(model_name=model_name)
        return self._models[model_name]

    # ------------------ LLM ------------------
    def get_llm_response(self, prompt):
        max_attempts_per_model = 2
        # Try known-good model first to avoid re-hitting deprecated models on each prompt.
        candidate_models = []
        if self.preferred_model and self.preferred_model in self.model_names:
            candidate_models.append(self.preferred_model)
        candidate_models.extend([m for m in self.model_names if m != self.preferred_model])

        for model_name in candidate_models:
            if model_name in self.disabled_models:
                continue
            for attempt in range(1, max_attempts_per_model + 1):
                try:
                    model = self._get_or_create_model(model_name)
                    response = model.generate_content(prompt)
                    response_text = getattr(response, "text", "")

                    if response_text and response_text.strip():
                        self.preferred_model = model_name
                        print(f"Using Gemini model: {model_name}")
                        return response_text.strip()

                    print(f"EMPTY LLM RESPONSE ({model_name}, attempt {attempt}/{max_attempts_per_model})")
                except Exception as e:
                    error_text = str(e)
                    print(f"LLM ERROR ({model_name}, attempt {attempt}/{max_attempts_per_model}): {error_text}")
                    # Permanently skip models that are unavailable/deprecated for this account.
                    lowered = error_text.lower()
                    if (
                        "no longer available" in lowered
                        or "is not found" in lowered
                        or "not supported for generatecontent" in lowered
                    ):
                        self.disabled_models.add(model_name)
                        break

                # Small backoff to avoid rate-limit/concurrency bursts
                time.sleep(0.5 * attempt)

        return ""

    # ------------------ EMBEDDINGS ------------------
    def get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    # ------------------ FILE HANDLING ------------------
    def extract_text_from_pdf(self, pdf_file):
        try:
            if hasattr(pdf_file, 'getvalue'):
                pdf_data = pdf_file.getvalue()
                reader = PdfReader(io.BytesIO(pdf_data))
            else:
                reader = PdfReader(pdf_file)

            text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content

            return text
        except Exception as e:
            print("PDF ERROR:", e)
            return ""

    def extract_text_from_file(self, file):
        ext = file.name.split('.')[-1].lower()
        if ext == "pdf":
            return self.extract_text_from_pdf(file)
        return ""

    # ------------------ VECTOR STORE ------------------
    def create_vector_store(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_text(text)

        print("Chunks created:", len(chunks))

        self.vectorstore = FAISS.from_texts(
            chunks,
            self.get_embeddings()
        )

    def retrieve_context(self, query, k=5):
        docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        print("\n🔍 RETRIEVED CONTEXT:\n", context[:300])
        return context

    def _extract_score(self, response_text):
        # Prefer explicit x/10 format first.
        match = re.search(r"\b(10|[0-9])\s*/\s*10\b", response_text)
        if match:
            return int(match.group(1))
        # Then fallback to first standalone 0-10 number.
        match = re.search(r"\b(10|[0-9])\b", response_text)
        return int(match.group(1)) if match else 0

    def _reason_from_response(self, response_text):
        parts = response_text.split(".", 1)
        return parts[1].strip() if len(parts) > 1 else response_text.strip() or "No reasoning"

    # ------------------ SKILL ANALYSIS ------------------
    def analyze_skill(self, skill):
        query = f"{skill} experience projects tools implementation usage"
        context = self.retrieve_context(query)

        if not context.strip():
            context = self.resume_text[:2000]

        prompt = f"""
You are a strict resume evaluator.

Context from resume:
{context}

Evaluate ONLY based on this context.

Task:
Check if the candidate demonstrates {skill}.
Target role: {self.current_role}
Candidate level expectation: {getattr(self, 'current_experience_level', 'mid')}

Rules:
- If clearly mentioned → 7-10
- If partially mentioned → 4-6
- If not mentioned → 0-3
- Evaluate relevance and evidence for the target role.

Output STRICTLY:
<number>. <reason>

Example:
8. Candidate used Python in ML projects.
"""

        response = self.get_llm_response(prompt)

        print("\n📊 SKILL:", skill)
        print("🧠 RESPONSE:", response)

        if not response:
            return skill, 0, "No response"

        # Prefer the first standalone score-like number (0-10), not years/dates.
        score = self._extract_score(response)
        reasoning = self._reason_from_response(response)

        return skill, max(0, min(score, 10)), reasoning

    def _score_section(self, section_name, section_skills, skill_scores, section_weight):
        scored = [skill_scores.get(skill, 0) for skill in section_skills]
        avg_score_10 = (sum(scored) / len(scored)) if scored else 0
        weighted_score_100 = avg_score_10 * 10 * section_weight
        return avg_score_10, weighted_score_100

    def _adjust_modern_ai_score(self, modern_score, core_score, production_score):
        """
        Soften penalties for missing modern-stack tools when fundamentals are strong.
        Recruiter logic: ML-heavy candidates without full LLM stack should be moderate,
        not severely penalized.
        """
        # If fundamentals are strong, enforce a moderate floor for modern stack.
        if (
            core_score >= self.MODERN_AI_CORE_STRONG_THRESHOLD
            and production_score >= self.MODERN_AI_PRODUCTION_STRONG_THRESHOLD
        ):
            return max(modern_score, self.MODERN_AI_FLOOR_STRONG)
        if core_score >= self.MODERN_AI_CORE_VERY_STRONG_THRESHOLD:
            return max(modern_score, self.MODERN_AI_FLOOR_VERY_STRONG)
        return modern_score

    def _apply_baseline_protection(self, section_breakdown):
        # Prevent framework/tools sections from collapsing unrealistically.
        core_like = [k for k in section_breakdown if "core" in k]
        for sec, data in section_breakdown.items():
            sec_name = sec.lower()
            if ("framework" in sec_name or "tools" in sec_name or "modern" in sec_name):
                ref_key = core_like[0] if core_like else None
                if ref_key:
                    ref_score = section_breakdown[ref_key]["score"]
                    floor = round(ref_score * 0.55, 2)
                    if data["score"] < floor:
                        data["score"] = floor

    def _tool_bonus(self, section_breakdown):
        bonus = 0.0
        for sec_name, sec_data in section_breakdown.items():
            if any(k in sec_name for k in ["framework", "tools", "modern", "containers", "cloud"]):
                values = list(sec_data["skills"].values())
                strong_count = sum(1 for v in values if v >= 7)
                if strong_count >= 3:
                    bonus += 2.5
                elif strong_count >= 1:
                    bonus += 1.0
        return min(8.0, round(bonus, 2))

    def _project_experience_signal_score(self):
        text = (self.resume_text or "").lower()
        signals = [
            "project", "deployed", "production", "pipeline", "intern", "experience",
            "implemented", "built", "designed", "end-to-end", "rag", "mlops",
        ]
        hits = sum(1 for signal in signals if signal in text)
        return min(10, max(3, hits))

    def _consistency_penalty(self, skill_scores, role_profile):
        if role_profile.get("consistency_rule") != "ml_framework_consistency":
            return 0, ""
        deep_learning = skill_scores.get("Deep Learning", 0)
        pytorch = skill_scores.get("PyTorch", 0)
        tensorflow = skill_scores.get("TensorFlow", 0)
        if deep_learning > 8 and pytorch < 3 and tensorflow < 3:
            # Keep this penalty meaningful but not overly punitive.
            return 6, "Deep Learning is very strong but framework evidence (PyTorch/TensorFlow) is minimal."
        return 0, ""

    def _build_recruiter_reasoning(self, section_breakdown, penalties, strengths, weaknesses):
        lines = []
        lines.append("Profile evaluation balances foundational capability, production readiness, and modern AI tooling.")
        for section_name, section_data in section_breakdown.items():
            lines.append(
                f"- {section_name.replace('_', ' ').title()}: {section_data['score']}/10 "
                f"(weighted contribution {section_data['weighted_contribution']:.1f}/100)."
            )
        if penalties:
            lines.append("Penalties applied for profile consistency gaps:")
            for p in penalties:
                lines.append(f"- {p}")
        if strengths:
            lines.append(f"Key strengths: {', '.join(strengths[:6])}.")
        if weaknesses:
            lines.append(f"Primary gaps: {', '.join(weaknesses[:6])}.")
            lines.append("Candidate has strong fundamentals but lacks exposure to some modern tools. With guidance, they can quickly adapt.")
        return "\n".join(lines)

    def _compute_trainability_score(self, section_breakdown, role_profile):
        buckets = role_profile.get("trainability_sections", [])
        if not buckets:
            return 0.0
        values = [section_breakdown.get(name, {}).get("score", 0) for name in buckets]
        return round(sum(values) / len(values), 2)

    def llm_evaluate(self, candidate_data, role, experience_level, structured_scores):
        """
        LLM judgment layer: interprets structured results without recalculating them.
        It may slightly adjust decision/score only when context suggests it.
        """
        base_score = int(structured_scores.get("final_score", 0))
        base_decision = self._decision_label(structured_scores.get("decision_band", self._decision_band(base_score)))
        payload = {
            "role": role,
            "experience_level": experience_level,
            "resume_summary": candidate_data.get("resume_summary", ""),
            "structured_scores": {
                "final_score": base_score,
                "base_decision": base_decision,
                "section_breakdown": structured_scores.get("section_breakdown", {}),
                "strengths": structured_scores.get("strengths", [])[:10],
                "weaknesses": structured_scores.get("weaknesses", [])[:10],
                "trainability_score": structured_scores.get("trainability_score", 0),
                "penalty_points": structured_scores.get("penalty_points", 0),
                "bonus_points": structured_scores.get("bonus_points", 0),
            },
        }

        prompt = f"""
You are a senior technical recruiter.
Role: {role}
Experience Level: {experience_level}

Use the structured evaluation below as the primary truth. Do NOT recalculate technical section scores.
Your task is to interpret and validate the outcome like a human recruiter.

Rules:
1) Respect structured data; adjust only if genuinely warranted by context.
2) For borderline scores (55-70), you MAY adjust one level up/down.
3) For extreme scores (<45 or >85), avoid drastic changes.
4) Be less harsh for freshers; account for trainability and project signals.
5) Treat missing tools as learning gaps, not automatic rejection criteria.

Return STRICT JSON:
{{
  "final_decision": "Selected|Shortlisted|Borderline|Rejected",
  "confidence": "Low|Medium|High",
  "adjusted_score": 0,
  "reasoning": "Detailed recruiter-style explanation",
  "key_strengths": ["..."],
  "key_gaps": ["..."],
  "improvement_suggestions": ["..."]
}}

Input:
{json.dumps(payload, indent=2)}
"""
        response = self.get_llm_response(prompt)
        fallback = {
            "final_decision": base_decision,
            "confidence": "Low",
            "adjusted_score": base_score,
            "reasoning": "LLM judgment unavailable. Structured decision retained.",
            "key_strengths": structured_scores.get("strengths", [])[:3],
            "key_gaps": structured_scores.get("weaknesses", [])[:3],
            "improvement_suggestions": [],
        }
        if not response:
            return fallback

        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:
            fallback["reasoning"] = cleaned[:400] if cleaned else fallback["reasoning"]
            fallback["confidence"] = "Medium"
            return fallback

        adjusted_score = int(parsed.get("adjusted_score", base_score))
        # bounded adjustment logic
        max_delta = 6 if 55 <= base_score <= 70 else 3
        if base_score < 45 or base_score > 85:
            max_delta = 2
        adjusted_score = max(base_score - max_delta, min(base_score + max_delta, adjusted_score))
        adjusted_score = max(0, min(100, adjusted_score))

        result = {
            "final_decision": parsed.get("final_decision", base_decision),
            "confidence": parsed.get("confidence", "Medium"),
            "adjusted_score": adjusted_score,
            "reasoning": parsed.get("reasoning", ""),
            "key_strengths": parsed.get("key_strengths", [])[:5],
            "key_gaps": parsed.get("key_gaps", [])[:5],
            "improvement_suggestions": parsed.get("improvement_suggestions", [])[:5],
        }

        # Optional fallback guard: low-confidence keeps structured decision.
        if str(result["confidence"]).lower() == "low":
            result["final_decision"] = base_decision
            result["adjusted_score"] = base_score

        return result

    def semantic_skill_analysis(self, skills):
        base_profile = self.ROLE_PROFILES.get(self.current_role, self.ROLE_PROFILES[self.DEFAULT_ROLE])
        level, years = self._experience_level()
        self.current_experience_level = level
        role_profile = self._adjust_role_weights_for_experience(base_profile, level)
        sections = role_profile["sections"]

        # Build skill list only from the active role + role requirements.
        base_skills = []
        for section_meta in sections.values():
            base_skills.extend(section_meta["skills"])
        merged_skills = list(dict.fromkeys(base_skills + (skills or [])))

        if not merged_skills:
            return {
                "overall_score": 0,
                "skill_scores": {},
                "missing_skills": [],
                "strengths": [],
                "selected": False,
                "section_breakdown": {},
                "detailed_weaknesses": [],
                "reasoning": "No skills available for evaluation.",
            }

        # Sequential scoring is slower but significantly more stable with LLM APIs.
        results = [self.analyze_skill(skill) for skill in merged_skills]

        skill_scores = {}
        skill_reasoning = {}
        for skill, score, reason in results:
            skill_scores[skill] = score
            skill_reasoning[skill] = reason

        # Context-aware override for project/practical section from resume evidence.
        project_signal_score = self._project_experience_signal_score()
        for section_name, section_meta in sections.items():
            if "project" in section_name or "case" in section_name:
                for practical_skill in section_meta["skills"]:
                    if skill_scores.get(practical_skill, 0) < project_signal_score:
                        skill_scores[practical_skill] = project_signal_score

        section_breakdown = {}
        weighted_total = 0.0
        for section_name, section_meta in sections.items():
            section_skills = section_meta["skills"]
            section_weight = section_meta["weight"]
            section_score_10, weighted_score_100 = self._score_section(
                section_name, section_skills, skill_scores, section_weight
            )
            section_breakdown[section_name] = {
                "score": round(section_score_10, 2),
                "weight": int(section_weight * 100),
                "weighted_contribution": round(weighted_score_100, 2),
                "skills": {s: skill_scores.get(s, 0) for s in section_skills},
            }
        self._apply_baseline_protection(section_breakdown)
        # recompute weighted total after baseline protection
        weighted_total = 0.0
        for section_name, section_meta in sections.items():
            weighted_total += section_breakdown[section_name]["score"] * 10 * section_meta["weight"]

        # Rebalance modern AI section only for ML-family roles.
        if self.current_role in {"AI/ML Engineer", "Data Scientist"} and "modern_ai_stack" in section_breakdown:
            core_score = section_breakdown.get("core_ml_nlp", {}).get("score", 0)
            production_score = section_breakdown.get("mlops_production", {}).get("score", 0)
            modern_score = section_breakdown["modern_ai_stack"]["score"]
            adjusted_modern_score = self._adjust_modern_ai_score(modern_score, core_score, production_score)
            if adjusted_modern_score != modern_score:
                old_contribution = section_breakdown["modern_ai_stack"]["weighted_contribution"]
                new_contribution = round(
                    adjusted_modern_score * 10 * sections["modern_ai_stack"]["weight"], 2
                )
                section_breakdown["modern_ai_stack"]["score"] = round(adjusted_modern_score, 2)
                section_breakdown["modern_ai_stack"]["weighted_contribution"] = new_contribution
                weighted_total += (new_contribution - old_contribution)

        penalty_points, penalty_reason = self._consistency_penalty(skill_scores, role_profile)
        penalty_points = min(self.MAX_TOTAL_PENALTY, penalty_points)
        bonus_points = self._tool_bonus(section_breakdown)
        final_score = max(0, min(100, int(round(weighted_total + bonus_points - penalty_points))))
        decision_band = self._decision_band(final_score)
        selected = decision_band in {"strong_hire", "shortlist"}

        strengths = [s for s, sc in skill_scores.items() if sc >= 7]
        missing = [s for s, sc in skill_scores.items() if sc <= 4]
        detailed_weaknesses = [
            {
                "skill": skill,
                "score": skill_scores.get(skill, 0),
                "detail": skill_reasoning.get(skill, "Insufficient evidence found in resume context."),
                "suggestions": [],
                "example": "",
            }
            for skill in missing
        ]

        penalties = [penalty_reason] if penalty_reason else []
        reasoning = self._build_recruiter_reasoning(
            section_breakdown=section_breakdown,
            penalties=penalties,
            strengths=strengths,
            weaknesses=missing,
        )
        trainability_score = self._compute_trainability_score(section_breakdown, role_profile)
        if trainability_score >= 7:
            reasoning += (
                f"\nCandidate shows high trainability despite gaps in modern frameworks "
                f"(trainability score: {trainability_score}/10)."
            )
        elif trainability_score >= 6:
            reasoning += f"\nCandidate shows moderate trainability (trainability score: {trainability_score}/10)."
        structured_output = {
            "final_score": final_score,
            "decision_band": decision_band,
            "section_breakdown": section_breakdown,
            "strengths": strengths,
            "weaknesses": missing,
            "trainability_score": trainability_score,
            "penalty_points": penalty_points,
            "bonus_points": bonus_points,
        }
        llm_evaluation = self.llm_evaluate(
            candidate_data={
                "resume_summary": (self.resume_text or "")[:1200],
            },
            role=self.current_role,
            experience_level=level,
            structured_scores=structured_output,
        )
        final_decision = llm_evaluation.get("final_decision", self._decision_label(decision_band))

        return {
            "mode": "ROLE_MODE",
            "overall_score": final_score,
            "final_score": final_score,
            "skill_scores": skill_scores,
            "missing_skills": missing,
            "strengths": strengths,
            "selected": selected,
            "decision_band": decision_band,
            "trainability_score": trainability_score,
            "experience_level": level,
            "experience_years": years,
            "bonus_points": bonus_points,
            "penalty_points": penalty_points,
            "section_breakdown": section_breakdown,
            "penalties": penalties,
            "detailed_weaknesses": detailed_weaknesses,
            "reasoning": reasoning,
            "evaluation_note": (
                f"Role-aware recruiter evaluation for {self.current_role}: "
                "only role-relevant skills and penalties are applied."
            ),
            "structured_score": structured_output,
            "llm_evaluation": llm_evaluation,
            "final_decision": final_decision,
            # backward compatibility for current UI wiring
            "qualitative_feedback": {
                "verdict": final_decision,
                "confidence": llm_evaluation.get("confidence", "Medium"),
                "summary": llm_evaluation.get("reasoning", ""),
                "recruiter_comments": llm_evaluation.get("key_strengths", []),
                "top_improvements": llm_evaluation.get("improvement_suggestions", []),
            },
        }

    # ------------------ MAIN ------------------
    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None, role_name=None):
        self.resume_text = self.extract_text_from_file(resume_file)
        if role_name:
            self.current_role = role_name

        print("\n📄 RESUME TEXT LENGTH:", len(self.resume_text))

        if not self.resume_text:
            return {"error": "Resume text extraction failed"}

        self.create_vector_store(self.resume_text)
        mode = get_evaluation_mode(custom_jd)
        if mode == "JD_MODE":
            jd_data = parse_jd(custom_jd)
            level, _ = self._experience_level()
            self.analysis_result = evaluate_jd_mode(
                agent=self,
                resume_text=self.resume_text,
                jd_data=jd_data,
                experience_level=level,
            )
        else:
            skills = role_requirements if role_requirements else []
            self.analysis_result = evaluate_role_mode(self, skills)

        return self.analysis_result

    # ------------------ QA ------------------
    def ask_question(self, question):
        context = self.retrieve_context(question)

        prompt = f"""
Answer ONLY from this resume context.

Context:
{context}

Question:
{question}
"""
        return self.get_llm_response(prompt)

    # ------------------ INTERVIEW ------------------
    def generate_interview_questions(self, types, difficulty, n):
        resume_excerpt = (self.resume_text or "")[:2500]
        role = self.current_role or "Target role"
        exp_level = getattr(self, "current_experience_level", "fresher")
        target_n = max(8, min(10, int(n)))

        prompt = f"""
You are a senior technical interviewer.
Generate interview questions STRICTLY based on this candidate resume and target role.

Target role: {role}
Candidate level: {exp_level}
Requested difficulty: {difficulty}
Total questions: {target_n}

Resume excerpt:
{resume_excerpt}

Rules:
- DO NOT generate generic DSA-only questions.
- DO NOT ignore candidate projects or tech stack.
- For fresher profiles, keep difficulty practical and fair.
- Mention technologies explicitly where relevant.
- No repetition, no filler.

Question distribution:
1) Project-Based (40%)
2) Role/JD-Based (30%)
3) Practical/System Design (20%)
4) DSA (10%, only 1-2 light questions)

Return STRICT JSON:
{{
  "Project-Based": ["q1", "q2", "..."],
  "Role/JD-Based": ["q1", "..."],
  "Practical/System Design": ["q1", "..."],
  "DSA": ["q1", "..."]
}}
"""
        res = self.get_llm_response(prompt)
        if not res:
            return []

        cleaned = res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            ordered_sections = [
                "Project-Based",
                "Role/JD-Based",
                "Practical/System Design",
                "DSA",
            ]
            output = []
            for section in ordered_sections:
                questions = parsed.get(section, [])
                if isinstance(questions, list):
                    for q in questions:
                        q_text = str(q).strip()
                        if q_text:
                            output.append((section, q_text))
            return output[:target_n]
        except Exception:
            # Fallback: keep compatibility with existing rendering.
            lines = [line.strip("- ").strip() for line in cleaned.split("\n") if line.strip()]
            return [("General", line) for line in lines[:target_n]]

    # ------------------ IMPROVE ------------------
    def improve_resume(self, improvement_areas, target_role=""):
        """
        Return structured resume improvement guidance by selected areas.
        Output format is a dict: {area: [suggestion1, suggestion2, ...]}.
        """
        if not self.resume_text:
            return {"General": ["Please analyze/upload a resume first so improvements can be personalized."]}

        areas = improvement_areas or ["Content", "Skills Highlighting"]
        prompt = f"""
You are a senior resume coach.
Improve this candidate resume for role: {target_role or "General software role"}.

Selected focus areas: {", ".join(areas)}

Resume:
{self.resume_text[:3500]}

Return STRICT JSON object where:
- keys are improvement areas
- values are arrays of concise, actionable suggestions (3-5 each)

Example:
{{
  "Content": ["...", "..."],
  "Projects": ["...", "..."]
}}
"""
        res = self.get_llm_response(prompt)
        if not res:
            return {area: [f"Add measurable impact and concrete examples for {area.lower()}."] for area in areas}

        cleaned = res.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                out = {}
                for area in areas:
                    val = parsed.get(area, [])
                    if isinstance(val, list):
                        out[area] = [str(x).strip() for x in val if str(x).strip()][:5]
                    elif isinstance(val, str) and val.strip():
                        out[area] = [val.strip()]
                    else:
                        out[area] = [f"Strengthen {area.lower()} with clearer, impact-focused evidence."]
                return out
        except Exception:
            pass

        # Fallback parse if LLM returns plain text
        fallback_lines = [line.strip("- ").strip() for line in cleaned.split("\n") if line.strip()]
        if not fallback_lines:
            fallback_lines = ["Add quantified impact, clear ownership, and stronger role alignment."]
        return {area: fallback_lines[:5] for area in areas}

    def get_improved_resume(self, target_role="", highlight_skills=""):
        prompt = f"""
Improve this resume for {target_role} role:

{self.resume_text}
"""
        return self.get_llm_response(prompt)