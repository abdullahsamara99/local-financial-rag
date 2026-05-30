# src/rag_pipeline.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate

class LocalRAGPipeline:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"⚙️ Loading model {model_id} via strict Chat Template configuration...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Clean and deterministic text generation pipeline
        self.hf_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=120,
            do_sample=False,
            return_full_text=False
        )

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        # Structure the retrieved context chunks
        context_str = "\n\n".join([f"[Page {doc['metadata'].get('page')}]: {doc['text']}" for doc in retrieved_docs])
        
        # Build a strict System & User message role layout
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict and precise financial assistant. Your job is to answer the user's question "
                    "using ONLY the provided retrieved context. \n"
                    "Strict Rules:\n"
                    "1. Match the language of the question. If the user asks in Arabic, answer in direct, fluent Arabic.\n"
                    "2. Output ONLY the factual answer in one concise sentence. Do not introduce it with headers.\n"
                    "3. Absolutely NO additional commentary, side-notes, disclaimers, or explanations. "
                    "Never use words like 'Note:' or 'Summary:'."
                )
            },
            {
                "role": "user",
                "content": f"Retrieved Context:\n{context_str}\n\nQuestion: {query}\nDirect Answer:"
            }
        ]
        
        # Format the messages using Qwen's specific chat template structure
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate answer while overriding the default max_length limits to prevent freezing
        output = self.hf_pipeline(prompt_formatted, max_length=None)
        return output[0]['generated_text'].strip()