"""
Dermatology Retrieval Assistant & Educational Chatbot Engine (src/chatbot.py)

Provides grounded, patient-friendly explanations of dermatological image retrieval
results, educational guidance on skin health, and dermatology concepts while strictly
enforcing safety constraints (no clinical diagnosis, no probability confusion).
"""

import os
import json
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

# ══════════════════════════════════════════════════════════════════════════════
# MEDICAL & DERMATOLOGY KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

DX_MAPPING = {
    "mel": {
        "name": "Melanoma",
        "category": "malignant",
        "description": "A serious form of skin cancer that begins in melanocytes (pigment-producing cells). Early detection and professional medical evaluation are crucial.",
    },
    "nv": {
        "name": "Melanocytic Nevus (Common Mole)",
        "category": "benign",
        "description": "A common benign (non-cancerous) skin growth formed by clusters of melanocytes.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "category": "malignant",
        "description": "The most common form of skin cancer, arising from basal cells in the lower layer of the epidermis. Typically slow-growing.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "category": "benign",
        "description": "Includes seborrheic keratoses, solar lentigines, and lichen planus-like keratoses. These are non-cancerous skin growths.",
    },
    "akiec": {
        "name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "category": "precancerous / malignant",
        "description": "A rough, scaly patch on the skin caused by years of sun exposure. Considered precancerous or early-stage carcinoma in situ.",
    },
    "vasc": {
        "name": "Vascular Lesion",
        "category": "benign",
        "description": "Includes cherry angiomas, angiokeratomas, and pyogenic granulomas. These involve blood vessels and are usually benign.",
    },
    "df": {
        "name": "Dermatofibroma",
        "category": "benign",
        "description": "A benign firm skin papule or nodule, often found on the lower legs.",
    },
}

ABCDE_RULE = {
    "A - Asymmetry": "One half of the lesion does not match the other half in shape, color, or pattern.",
    "B - Border": "The edges are irregular, ragged, notched, blurred, or poorly defined.",
    "C - Color": "Color is not uniform; may include varying shades of brown, black, pink, red, white, or blue.",
    "D - Diameter": "The lesion is larger than 6mm (about the size of a pencil eraser), though melanomas can be smaller.",
    "E - Evolving": "The lesion changes in size, shape, color, or elevation, or presents new symptoms like itching or bleeding.",
}

SYSTEM_PROMPT = """You are a specialized Dermatology Educational Assistant and Image Retrieval Interpreter.
Your task is to interpret visual image-retrieval evidence, explain dermatological terms, provide evidence-based patient education, and guide users on safe next steps.

CRITICAL SAFETY DIRECTIVES:
1. NEVER provide a clinical diagnosis or state "Your lesion is X" or "You have skin cancer".
2. NEVER equate vector similarity (e.g. 0.92 cosine similarity) with diagnostic probability (it is NOT a 92% chance of cancer). Explain that similarity reflects visual feature closeness in AI feature space.
3. ALWAYS include a clear medical disclaimer reminding the user that this tool is for educational/research purposes only and cannot replace an evaluation by a board-certified dermatologist.
4. Maintain an empathetic, professional, clear, and objective tone.
"""


# ══════════════════════════════════════════════════════════════════════════════
# DERMATOLOGY CHATBOT CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DermChatbot:
    """Retrieval-augmented chatbot for explaining skin lesion search results and answering questions."""

    def __init__(self, provider: str = "Built-in Grounded Assistant", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key

    def format_retrieval_summary(self, results_df: pd.DataFrame) -> str:
        """Converts retrieved DataFrame into a readable summary string for LLM context."""
        if results_df is None or len(results_df) == 0:
            return "No retrieval results available."

        total = len(results_df)
        categories = results_df["category"].value_counts().to_dict() if "category" in results_df else {}
        dx_counts = results_df["dx"].value_counts().to_dict() if "dx" in results_df else {}

        summary_lines = [f"Retrieved Top-{total} Visually Similar Cases from Database:"]
        
        for idx, row in results_df.iterrows():
            img_id = row.get("image_id", f"Case {idx+1}")
            dx_code = str(row.get("dx", "unknown")).lower()
            dx_info = DX_MAPPING.get(dx_code, {"name": dx_code.upper(), "category": row.get("category", "unknown")})
            sim = row.get("similarity", 0.0)
            summary_lines.append(
                f"- Case #{idx+1} ({img_id}): Diagnosis = {dx_info['name']} ({dx_info['category']}), Similarity Score = {sim:.2f}"
            )

        cat_summary = ", ".join([f"{count} {cat}" for cat, count in categories.items()])
        summary_lines.append(f"Summary distribution: {cat_summary}.")
        return "\n".join(summary_lines)

    def generate_built_in_response(
        self, user_query: str, retrieval_df: Optional[pd.DataFrame] = None
    ) -> str:
        """Rule-augmented grounded fallback engine when no external API key is provided."""
        query_lower = user_query.lower()
        response_parts = []

        # 1. Handle Retrieval Result Explanation
        if any(w in query_lower for w in ["explain", "result", "search", "retrieved", "matches", "summary", "find"]):
            if retrieval_df is not None and len(retrieval_df) > 0:
                summary_text = self.format_retrieval_summary(retrieval_df)
                total = len(retrieval_df)
                cats = retrieval_df["category"].value_counts().to_dict() if "category" in retrieval_df else {}
                
                response_parts.append(
                    f"### 📊 Visual Retrieval Results Breakdown\n\n"
                    f"Based on feature extraction from your uploaded lesion image, our deep learning model retrieved the top **{total} most visually similar cases** from the ISIC database:\n\n"
                    f"{summary_text}\n\n"
                    f"**What does this mean?**\n"
                    f"- The model looks at dermatoscopic visual patterns (such as color distribution, border sharpness, and pigment network structures).\n"
                    f"- A similarity score of **{retrieval_df['similarity'].iloc[0]:.2f}** indicates strong similarity in deep visual feature space to database case `{retrieval_df['image_id'].iloc[0]}`.\n"
                    f"- **Important**: A high visual similarity score measures how much two images resemble each other visually — **it is NOT a probability calculation of disease or cancer**."
                )
            else:
                response_parts.append(
                    "Please upload a skin lesion image and run a search first! Once search results are generated, I can explain the visual matches, diagnosis breakdown, and similarity metrics for you."
                )

        # 2. Handle ABCDE Rule Query
        elif any(w in query_lower for w in ["abcde", "border", "asymmetry", "color", "diameter", "evolving", "rule"]):
            response_parts.append(
                "### 🩺 The ABCDE Rule for Skin Lesion Self-Examination\n\n"
                "Dermatologists use the **ABCDE criteria** as a guide to help identify warning signs of melanoma:\n\n"
            )
            for key, desc in ABCDE_RULE.items():
                response_parts.append(f"- **{key}**: {desc}\n")
            response_parts.append(
                "\n*If you notice any lesion displaying these features or evolving over time, it is recommended to have it evaluated by a dermatologist.*"
            )

        # 3. Handle Similarity vs Probability / Cancer question
        elif any(w in query_lower for w in ["probability", "percent", "cancer", "diagnose", "diagnosis", "accuracy", "meaning"]):
            response_parts.append(
                "### 🔬 Understanding Visual Similarity vs. Clinical Diagnosis\n\n"
                "**1. What is Cosine Similarity?**\n"
                "In Content-Based Image Retrieval (CBIR), similarity scores (e.g. 0.92 or 92%) reflect how close two images are in a high-dimensional mathematical feature space (extracted by neural networks like EfficientNet).\n\n"
                "**2. Why Similarity is NOT Cancer Probability:**\n"
                "- A 90% similarity score means: *'This lesion shares 90% of its visual feature representation with a database image.'*\n"
                "- It does **NOT** mean: *'There is a 90% chance this lesion is malignant.'*\n\n"
                "**3. Why Clinical Evaluation Matters:**\n"
                "Dermatologists combine visual appearance with dermoscopy, clinical history, patient risk factors, and often biopsy analysis to make a diagnosis."
            )

        # 4. Handle Next Steps / Guidance
        elif any(w in query_lower for w in ["next", "should i do", "guidance", "doctor", "dermatologist", "what to do", "action"]):
            response_parts.append(
                "### 🩺 Recommended Next Steps & Guidance\n\n"
                "1. **Monitor for Changes**: Use the ABCDE rule to track any evolving changes in size, shape, color, or sensation (itching/bleeding).\n"
                "2. **Schedule a Professional Skin Exam**: If you have concerns about a new or changing spot, book an appointment with a board-certified dermatologist.\n"
                "3. **Prepare for Your Visit**: Note when you first noticed the lesion and any changes over time.\n"
                "4. **Practice Sun Safety**: Use broad-spectrum sunscreen (SPF 30+), seek shade during peak hours, and wear protective clothing."
            )

        # 5. Default General Educational Response
        else:
            response_parts.append(
                "### 💡 Skin Health & Image Retrieval Assistant\n\n"
                "I am here to help you understand your image retrieval search results and skin health concepts. Here are key things I can assist with:\n\n"
                "- 📊 **Explain Search Results**: Breakdown of top retrieved cases and diagnosis distributions.\n"
                "- 🩺 **Dermatology Education**: Learn about the ABCDE rule, lesion types (Melanoma, Nevi, BCC), and risk factors.\n"
                "- 🔬 **AI Concepts**: Understand how deep learning embeddings and FAISS vector similarity work.\n"
                "- ⚠️ **Next Steps**: Advice on preparing for professional dermatological skin exams.\n\n"
                "*Feel free to ask a specific question or click one of the quick action buttons above!*"
            )

        # Add mandatory medical disclaimer
        response_parts.append(
            "\n\n---\n> **⚠️ Medical Disclaimer**: This system is designed solely for educational and research purposes. It does not perform medical diagnosis, prescribe treatment, or replace professional dermatological care."
        )

        return "".join(response_parts)

    def generate_openai_response(
        self, user_query: str, chat_history: List[Dict[str, str]], retrieval_df: Optional[pd.DataFrame] = None
    ) -> str:
        """Invokes OpenAI ChatGPT API if key is provided."""
        if not self.api_key:
            return "OpenAI API Key is missing. Please enter your API key in the sidebar or switch to the Built-in Grounded Assistant."

        try:
            context_str = self.format_retrieval_summary(retrieval_df) if retrieval_df is not None else "No image uploaded yet."
            
            messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCurrent Search Context:\n{context_str}"}]
            for msg in chat_history[-6:]:  # include recent turns
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_query})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 600,
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error contacting OpenAI API: {e}\n\nFalling back to built-in response:\n\n" + self.generate_built_in_response(user_query, retrieval_df)

    def generate_gemini_response(
        self, user_query: str, chat_history: List[Dict[str, str]], retrieval_df: Optional[pd.DataFrame] = None
    ) -> str:
        """Invokes Google Gemini API if key is provided."""
        if not self.api_key:
            return "Google Gemini API Key is missing. Please enter your API key in the sidebar or switch to the Built-in Grounded Assistant."

        try:
            context_str = self.format_retrieval_summary(retrieval_df) if retrieval_df is not None else "No image uploaded yet."
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            
            prompt_text = f"{SYSTEM_PROMPT}\n\nCurrent Search Context:\n{context_str}\n\nUser Question: {user_query}"
            payload = {"contents": [{"parts": [{"text": prompt_text}]}]}

            resp = requests.post(url, json=payload, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Error contacting Gemini API: {e}\n\nFalling back to built-in response:\n\n" + self.generate_built_in_response(user_query, retrieval_df)

    def respond(
        self, user_query: str, chat_history: List[Dict[str, str]], retrieval_df: Optional[pd.DataFrame] = None
    ) -> str:
        """Main entrypoint for generating responses based on selected provider."""
        if self.provider == "OpenAI (GPT-4o-mini)" and self.api_key:
            return self.generate_openai_response(user_query, chat_history, retrieval_df)
        elif self.provider == "Google Gemini" and self.api_key:
            return self.generate_gemini_response(user_query, chat_history, retrieval_df)
        else:
            return self.generate_built_in_response(user_query, retrieval_df)
