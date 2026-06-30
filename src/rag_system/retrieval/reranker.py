from typing import Any, List
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: Optional[Any]

class BGEReranker:
    def __init__(self):
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder("BAAI/bge-reranker-base")
            except Exception:
                self.model = None
        else:
            self.model = None

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int = 4,
    ) -> List[Any]:
        if not documents:
            return []

        # Enforce Mock scores for testing / fallback if no model is loaded
        if self.model is None:
            for i, doc in enumerate(documents):
                if hasattr(doc, "metadata") and doc.metadata is not None:
                    doc.metadata["score"] = 1.0 - (i * 0.1)
            return documents[:top_k]

        pairs = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                pairs.append((query, doc.page_content))
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                pairs.append((query, doc.chunk.text))
            elif isinstance(doc, dict):
                pairs.append((query, doc.get("text", "")))
            else:
                pairs.append((query, str(doc)))

        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            if hasattr(doc, "metadata") and doc.metadata is not None:
                doc.metadata["score"] = float(score)
            elif hasattr(doc, "score"):
                try:
                    object.__setattr__(doc, "score", float(score))
                except Exception:
                    pass

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            doc
            for doc, _
            in ranked[:top_k]
        ]
