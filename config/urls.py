"""Root URL configuration for the Personalised RAG project."""

from django.contrib import admin
from django.urls import include, path

from rag_app import views

urlpatterns = [
    path("", views.index, name="index"),
    path("admin/", admin.site.urls),
    path("api/", include(("rag_app.urls", "rag_app"), namespace="rag_app")),
]
