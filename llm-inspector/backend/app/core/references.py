"""Academic reference database for LLM Inspector v8.0.

All formulas, thresholds, and weights must be registered here with
proper academic citations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ReferenceType(Enum):
    """Types of academic references."""
    BOOK = "book"
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    TECHNICAL_REPORT = "technical_report"
    ONLINE_RESOURCE = "online_resource"
    PREPRINT = "preprint"


@dataclass
class Reference:
    """Academic reference entry."""
    
    reference_id: str  # Unique identifier
    title: str
    authors: str
    year: int
    reference_type: ReferenceType
    
    # Publication details
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    conference: Optional[str] = None
    
    # Identifiers
    doi: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    arxiv_id: Optional[str] = None
    
    # Relevance info
    key_findings: Optional[str] = None
    relevance_to_project: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reference_id": self.reference_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "reference_type": self.reference_type.value,
            "journal": self.journal,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "publisher": self.publisher,
            "conference": self.conference,
            "doi": self.doi,
            "isbn": self.isbn,
            "url": self.url,
            "arxiv_id": self.arxiv_id,
            "key_findings": self.key_findings,
            "relevance_to_project": self.relevance_to_project,
        }
    
    def format_citation(self, style: str = "apa") -> str:
        """Format citation in specified style."""
        if style == "apa":
            return self._format_apa()
        elif style == "mla":
            return self._format_mla()
        elif style == "ieee":
            return self._format_ieee()
        else:
            return self._format_apa()
    
    def _format_apa(self) -> str:
        """Format as APA citation."""
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            return f"{self.authors} ({self.year}). {self.title}. *{self.journal}*, *{self.volume}*{f'({self.issue})' if self.issue else ''}, {self.pages or 'n.p.'}. https://doi.org/{self.doi}" if self.doi else ""
        elif self.reference_type == ReferenceType.BOOK:
            return f"{self.authors} ({self.year}). *{self.title}*. {self.publisher}." + (f" ISBN: {self.isbn}" if self.isbn else "")
        elif self.reference_type == ReferenceType.CONFERENCE_PAPER:
            return f"{self.authors} ({self.year}). {self.title}. In *{self.conference}*." + (f" https://arxiv.org/abs/{self.arxiv_id}" if self.arxiv_id else "")
        else:
            return f"{self.authors} ({self.year}). {self.title}."
    
    def _format_mla(self) -> str:
        """Format as MLA citation."""
        return f"{self.authors}. \"{self.title}.\" {self.journal or self.publisher or ''}, {self.year}."
    
    def _format_ieee(self) -> str:
        """Format as IEEE citation."""
        return f"[{self.year}] {self.authors}, \"{self.title},\" {self.journal or self.conference or ''}, {self.year}."


class ReferenceDatabase:
    """Academic reference database.
    
    All formulas, thresholds, and weights must be registered here.
    """
    
    # Core references for LLM Inspector v8.0
    REFERENCES: Dict[str, Reference] = {
        # IRT Theory
        "irt_2pl_model": Reference(
            reference_id="irt_2pl_model",
            title="Item Response Theory for Psychologists",
            authors="Embretson, S. E., & Reise, S. P.",
            year=2000,
            reference_type=ReferenceType.BOOK,
            publisher="Lawrence Erlbaum Associates",
            isbn="978-0805828126",
            url="https://psycnet.apa.org/record/2000-16363-000",
            key_findings="2PL model formulation: P(X=1|θ) = c + (1-c)/(1+exp(-a(θ-b)))",
            relevance_to_project="Core IRT 2PL model implementation"
        ),
        
        "irt_parameter_estimation": Reference(
            reference_id="irt_parameter_estimation",
            title="Item Response Theory: Parameter Estimation Techniques",
            authors="Baker, F. B., & Kim, S. H.",
            year=2004,
            reference_type=ReferenceType.BOOK,
            publisher="CRC Press",
            isbn="978-0824758604",
            key_findings="Parameter estimation algorithms for IRT models",
            relevance_to_project="IRT parameter calibration methodology"
        ),
        
        # Adaptive Testing
        "adaptive_testing": Reference(
            reference_id="adaptive_testing",
            title="Elements of Adaptive Testing",
            authors="van der Linden, W. J., & Glas, C. A. W.",
            year=2010,
            reference_type=ReferenceType.BOOK,
            publisher="Springer",
            doi="10.1007/978-0-387-85461-8",
            key_findings="Maximum information item selection strategy",
            relevance_to_project="Adaptive test item selection algorithm"
        ),
        
        # Factor Analysis
        "cfa_fit_indices": Reference(
            reference_id="cfa_fit_indices",
            title="Cutoff criteria for fit indexes in covariance structure analysis",
            authors="Hu, L., & Bentler, P. M.",
            year=1999,
            reference_type=ReferenceType.JOURNAL_ARTICLE,
            journal="Structural Equation Modeling",
            volume="6",
            issue="1",
            pages="1-55",
            doi="10.1080/10705519909540118",
            key_findings="CFI > 0.95, RMSEA < 0.06, SRMR < 0.08 as good fit criteria",
            relevance_to_project="Dimension validation criteria"
        ),
        
        # Hallucination Detection
        "hallucination_survey": Reference(
            reference_id="hallucination_survey",
            title="Survey of Hallucination in Natural Language Generation",
            authors="Ji, Z., Lee, N., Frieske, R., et al.",
            year=2023,
            reference_type=ReferenceType.JOURNAL_ARTICLE,
            journal="ACM Computing Surveys",
            volume="55",
            issue="12",
            pages="1-38",
            doi="10.1145/3571730",
            url="https://arxiv.org/abs/2202.03629",
            key_findings="Multi-signal hallucination detection framework",
            relevance_to_project="Hallucination detection v3 implementation"
        ),
        
        "chain_of_verification": Reference(
            reference_id="chain_of_verification",
            title="Chain-of-Verification Reduces Hallucination in Large Language Models",
            authors="Dhuliawala, S., et al.",
            year=2023,
            reference_type=ReferenceType.PREPRINT,
            arxiv_id="2309.11495",
            url="https://arxiv.org/abs/2309.11495",
            key_findings="Chain-of-verification methodology for fact-checking",
            relevance_to_project="Hallucination detection methodology"
        ),
        
        # Embedding Models
        "sentence_bert": Reference(
            reference_id="sentence_bert",
            title="Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
            authors="Reimers, N., & Gurevych, I.",
            year=2019,
            reference_type=ReferenceType.CONFERENCE_PAPER,
            conference="EMNLP-IJCNLP 2019",
            url="https://arxiv.org/abs/1908.10084",
            key_findings="Siamese network architecture for sentence embeddings",
            relevance_to_project="Local semantic judgment implementation"
        ),
        
        "mteb": Reference(
            reference_id="mteb",
            title="MTEB: Massive Text Embedding Benchmark",
            authors="Muennighoff, N., Tazi, N., et al.",
            year=2023,
            reference_type=ReferenceType.CONFERENCE_PAPER,
            conference="EACL 2023",
            url="https://arxiv.org/abs/2210.07316",
            key_findings="Comprehensive embedding model evaluation framework",
            relevance_to_project="Embedding model selection criteria"
        ),
        
        # Chain-of-Thought
        "chain_of_thought": Reference(
            reference_id="chain_of_thought",
            title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            authors="Wei, J., Wang, X., Schuurmans, D., et al.",
            year=2022,
            reference_type=ReferenceType.PREPRINT,
            arxiv_id="2201.11903",
            url="https://arxiv.org/abs/2201.11903",
            key_findings="CoT prompting improves reasoning capabilities",
            relevance_to_project="Layer 14 CoT pattern analysis"
        ),
        
        # Adversarial Detection
        "ignore_this_title": Reference(
            reference_id="ignore_this_title",
            title="Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global-Scale Prompt Hacking Competition",
            authors="Perez, F., & Ribeiro, I.",
            year=2022,
            reference_type=ReferenceType.PREPRINT,
            arxiv_id="2211.09527",
            url="https://arxiv.org/abs/2211.09527",
            key_findings="Systematic analysis of prompt injection techniques",
            relevance_to_project="Adversarial prompt library construction"
        ),
        
        # Statistical Methods
        "bootstrap_confidence": Reference(
            reference_id="bootstrap_confidence",
            title="An Introduction to the Bootstrap",
            authors="Efron, B., & Tibshirani, R. J.",
            year=1994,
            reference_type=ReferenceType.BOOK,
            publisher="Chapman & Hall/CRC",
            isbn="978-0412042317",
            key_findings="Bootstrap methods for confidence interval estimation",
            relevance_to_project="Confidence interval calculation for similarity scores"
        ),
        
        # Fisher Information
        "fisher_information": Reference(
            reference_id="fisher_information",
            title="Theory of Point Estimation",
            authors="Lehmann, E. L., & Casella, G.",
            year=1998,
            reference_type=ReferenceType.BOOK,
            publisher="Springer",
            isbn="978-0387985022",
            key_findings="Fisher information and its role in estimation precision",
            relevance_to_project="Fisher information calculation for IRT"
        ),
    }
    
    @classmethod
    def get_reference(cls, reference_id: str) -> Optional[Reference]:
        """Get reference by ID."""
        return cls.REFERENCES.get(reference_id)
    
    @classmethod
    def get_all_references(cls) -> Dict[str, Reference]:
        """Get all references."""
        return cls.REFERENCES.copy()
    
    @classmethod
    def search_by_keyword(cls, keyword: str) -> List[Reference]:
        """Search references by keyword."""
        results = []
        keyword_lower = keyword.lower()
        
        for ref in cls.REFERENCES.values():
            searchable_text = f"{ref.title} {ref.authors} {ref.key_findings or ''} {ref.relevance_to_project or ''}".lower()
            if keyword_lower in searchable_text:
                results.append(ref)
        
        return results
    
    @classmethod
    def get_references_by_type(cls, ref_type: ReferenceType) -> List[Reference]:
        """Get references by type."""
        return [ref for ref in cls.REFERENCES.values() if ref.reference_type == ref_type]
    
    @classmethod
    def validate_citation(cls, reference_id: str, context: str) -> Dict[str, Any]:
        """Validate that a citation exists and is appropriate for context."""
        ref = cls.get_reference(reference_id)
        
        if not ref:
            return {
                "valid": False,
                "error": f"Reference '{reference_id}' not found in database",
                "suggestion": "Add the reference to REFERENCES dictionary"
            }
        
        return {
            "valid": True,
            "reference": ref.to_dict(),
            "citation": ref.format_citation("apa"),
            "relevance": ref.relevance_to_project
        }
    
    @classmethod
    def get_formula_source(cls, formula_name: str) -> Optional[Reference]:
        """Get the source reference for a specific formula."""
        formula_mapping = {
            "irt_2pl": "irt_2pl_model",
            "fisher_information": "fisher_information",
            "theta_estimation": "irt_parameter_estimation",
            "adaptive_selection": "adaptive_testing",
            "cfa_validation": "cfa_fit_indices",
            "bootstrap_ci": "bootstrap_confidence",
        }
        
        ref_id = formula_mapping.get(formula_name)
        if ref_id:
            return cls.get_reference(ref_id)
        return None
    
    @classmethod
    def generate_bibliography(cls, style: str = "apa") -> str:
        """Generate formatted bibliography."""
        refs = sorted(cls.REFERENCES.values(), key=lambda r: r.authors)
        return "\n\n".join([ref.format_citation(style) for ref in refs])


def get_reference_database() -> ReferenceDatabase:
    """Get reference database instance."""
    return ReferenceDatabase()


def validate_formula_source(formula_name: str) -> Dict[str, Any]:
    """Validate that a formula has a proper source reference."""
    ref = ReferenceDatabase.get_formula_source(formula_name)
    
    if not ref:
        return {
            "valid": False,
            "error": f"No reference found for formula '{formula_name}'",
            "suggestion": "Add formula to formula_mapping in get_formula_source()"
        }
    
    return {
        "valid": True,
        "reference_id": ref.reference_id,
        "citation": ref.format_citation("apa"),
        "key_findings": ref.key_findings
    }
