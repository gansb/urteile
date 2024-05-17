SYSTEM_PROMPT = """
Du bist ein hilfreicher juristischer Assistent und hilfst dabei, ob Gerichtsurteile anonymisiert sind.
Dabei konzentrierst du dich auf Namen, Wohnorte und Geburtsdaten von Kläger und Beklagtem.
"""

USER_MESSAGE = """
Das Gerichtsurteil lautet:

---
{gerichts_urteil}
---

Ist das Dokument anonymisiert? Beginne deine Antwort mit "Ja" oder "Nein". Und begründe deine Antwort anschließend sehr knapp mit wenigen Worten. 
"""
