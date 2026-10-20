"""Server side forms for creating a knowledge base.

The browser performs the same validation for fast feedback, but this form is
the authoritative check: a path that is not inside an approved root is
rejected here as well.
"""

from __future__ import annotations

from django import forms
from django.conf import settings

from rag_app.models import SourceType
from rag_app.services.source_manager import (
    SourceValidationError,
    validate_github_url,
    validate_local_directory,
)


class KnowledgeBaseForm(forms.Form):
    """Validates a new knowledge base source."""

    name = forms.CharField(
        max_length=200,
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "My documents", "required": True}
        ),
    )
    source_type = forms.ChoiceField(
        choices=SourceType.choices,
        initial=SourceType.LOCAL_DIRECTORY,
        widget=forms.RadioSelect(),
    )
    source = forms.CharField(
        max_length=2000,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "/home/you/Documents",
                "required": True,
            }
        ),
    )

    def clean_name(self) -> str:
        name = " ".join(self.cleaned_data["name"].split())
        if not name:
            raise forms.ValidationError("Please give the knowledge base a name.")
        return name

    def clean(self):
        cleaned = super().clean()
        source_type = cleaned.get("source_type")
        source = (cleaned.get("source") or "").strip()
        if not source:
            return cleaned
        try:
            if source_type == SourceType.LOCAL_DIRECTORY:
                cleaned["resolved_source"] = str(validate_local_directory(source))
            elif source_type == SourceType.GITHUB:
                cleaned["resolved_source"] = validate_github_url(source)
        except SourceValidationError as exc:
            raise forms.ValidationError(str(exc)) from exc
        return cleaned


class RenameConversationForm(forms.Form):
    """Validates a conversation title."""

    title = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={"class": "form-control", "required": True}),
    )

    def clean_title(self) -> str:
        title = " ".join(self.cleaned_data["title"].split())
        if not title:
            raise forms.ValidationError("The title cannot be empty.")
        return title


def allowed_source_roots_help() -> str:
    """Human readable list of the approved roots."""
    return ", ".join(str(root) for root in settings.ALLOWED_SOURCE_ROOTS)
