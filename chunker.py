import os
import glob
import re
from typing import List, Dict

class DocumentChunker:
    @staticmethod
    def chunk_guidelines(guidelines_dir: str) -> List[Dict]:
        chunks = []
        files = glob.glob(os.path.join(guidelines_dir, "*.md"))
        
        for file_path in files:
            disease = os.path.basename(file_path).replace(".md", "").capitalize()
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Split by section headers
            sections = re.split(r"(## SECTION \d+: .*?)\n", content)
            
            title = "General"
            if len(sections) > 1:
                # First element is usually the file title/intro
                intro = sections[0].strip()
                if intro:
                    chunks.append({
                        "disease": disease,
                        "section": "Introduction",
                        "content": intro,
                        "citation": f"{disease} Guideline Intro"
                    })
                
                # Pair the section headers with their content
                for i in range(1, len(sections), 2):
                    sec_header = sections[i].strip()
                    sec_content = sections[i+1].strip() if i+1 < len(sections) else ""
                    
                    # Remove markdown horizontal rules
                    sec_content = re.sub(r"\n---\n", "\n", sec_content)
                    
                    chunks.append({
                        "disease": disease,
                        "section": sec_header,
                        "content": sec_content,
                        "citation": f"{disease} Guideline - {sec_header}"
                    })
            else:
                # If no clear sections, split by paragraphs
                paragraphs = content.split("\n\n")
                for idx, p in enumerate(paragraphs):
                    if p.strip():
                        chunks.append({
                            "disease": disease,
                            "section": f"Paragraph {idx+1}",
                            "content": p.strip(),
                            "citation": f"{disease} Guideline"
                        })
                        
        return chunks

if __name__ == "__main__":
    import json
    chunks = DocumentChunker.chunk_guidelines("data/guidelines")
    print(f"Generated {len(chunks)} chunks.")
    print(json.dumps(chunks[1], indent=2))
