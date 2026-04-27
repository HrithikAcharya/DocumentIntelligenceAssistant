"""
Citation parser for the Document Intelligence Assistant.
Extracts [Source: FileName.pdf, Page X] citations from LLM responses
and renders them as clickable Chainlit action deep links.
"""
import re
from dataclasses import dataclass
from typing import Optional

import chainlit as cl

# Regex pattern matching [Source: FileName.pdf, Page X]
CITATION_PATTERN = re.compile(r'\[Source:\s*(.+?\.pdf),\s*Page\s*(\d+)\]')


@dataclass
class Citation:
    """A parsed citation referencing a specific page in a PDF document."""
    filename: str   # e.g., "report.pdf"
    page: int       # 1-indexed page number


class CitationParser:
    """
    Extracts citations from LLM response text and renders them as
    Chainlit action deep links for PDF viewer navigation.
    """

    def extract_citations(self, text: str) -> list[Citation]:
        """
        Extract all citations from response text.

        Args:
            text: LLM response text potentially containing citation patterns.

        Returns:
            List of Citation objects parsed from the text.
            Returns empty list if no citations found.
        """
        citations = []
        for match in CITATION_PATTERN.finditer(text):
            filename = match.group(1).strip()
            page = int(match.group(2))
            if page > 0:  # Only include valid positive page numbers
                citations.append(Citation(filename=filename, page=page))
        return citations

    def render_as_actions(
        self,
        text: str,
        citations: list[Citation],
    ) -> tuple[str, list[cl.Action]]:
        """
        Replace citation patterns in text with markdown link placeholders
        and create corresponding Chainlit Action objects for deep linking.

        Args:
            text: LLM response text containing citation patterns.
            citations: List of Citation objects extracted from the text.

        Returns:
            Tuple of (modified_text, list_of_cl.Action objects).
            The modified text has citations replaced with clickable markdown links.
            Each Action navigates the PDF viewer to the cited page.
        """
        actions = []
        seen_values = set()

        def replace_citation(match: re.Match) -> str:
            filename = match.group(1).strip()
            page = int(match.group(2))
            action_value = f"{filename}:{page}"
            label = f"📄 {filename}, Page {page}"

            # Create one action per unique citation
            if action_value not in seen_values:
                seen_values.add(action_value)
                actions.append(
                    cl.Action(
                        name="navigate_pdf",
                        value=action_value,
                        label=label,
                        description=f"Navigate to {filename}, page {page}",
                    )
                )

            # Replace with a styled inline reference
            return f"**[{filename}, p.{page}]**"

        modified_text = CITATION_PATTERN.sub(replace_citation, text)
        return modified_text, actions
