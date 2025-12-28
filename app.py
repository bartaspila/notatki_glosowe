from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
import datetime

# ---------------------
# CONFIG
# ---------------------
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
QDRANT_COLLECTION_NAME = "notes"

# ---------------------
# CLIENTS
# ---------------------
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

# ---------------------
# DB SETUP
# ---------------------
def assure_db_collection_exists():
    client = get_qdrant_client()
    try:
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            st.info("Utworzono kolekcję Qdrant.")
        else:
            st.info("Kolekcja Qdrant istnieje.")
    except Exception as e:
        st.error(f"Błąd przy sprawdzaniu kolekcji Qdrant: {e}")

# ---------------------
# OPENAI FUNCTIONS
# ---------------------
def get_embedding(text):
    text = (text or "")[:8000]
    if not text.strip():
        st.warning("Nie można utworzyć embeddingu z pustego tekstu.")
        return None
    openai_client = get_openai_client()
    try:
        result = openai_client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return result.data[0].embedding
    except Exception as e:
        st.error(f"Błąd podczas tworzenia embeddingu: {e}")
        return None

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    try:
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=AUDIO_TRANSCRIBE_MODEL
        )
        return transcript.text
    except Exception as e:
        st.error(f"Błąd transkrypcji audio: {e}")
        return ""

# ---------------------
# QDRANT FUNCTIONS
# ---------------------
def add_note_to_db(note_text):
    embedding = get_embedding(note_text)
    if not embedding or len(embedding) != EMBEDDING_DIM:
        st.error("Nie udało się utworzyć embeddingu dla notatki.")
        return

    try:
        client = get_qdrant_client()
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": note_text,
                        "created_at": datetime.datetime.utcnow().isoformat()
                    }
                )
            ]
        )
        st.success("Notatka zapisana 🎉")
    except Exception as e:
        st.error(f"Błąd przy zapisie notatki do Qdrant: {e}")

def list_notes_from_db(query=None):
    client = get_qdrant_client()
    points = []

    if not query:
        try:
            response = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=100,
                with_payload=True
            )
            points = response.result
        except Exception as e:
            st.error(f"Błąd pobierania notatek: {e}")
    else:
        embedding = get_embedding(query)
        if embedding:
            try:
                points = client.search(
                    collection_name=QDRANT_COLLECTION_NAME,
                    vector=embedding,
                    limit=10,
                    with_payload=True
                )
            except Exception as e:
                st.error(f"Błąd przy wyszukiwaniu: {e}")

    # sortuj po dacie jeśli jest
    notes = []
    for note in points:
        text = note.payload.get("text") if note.payload else ""
        created_at = note.payload.get("created_at") if note.payload else ""
        score = getattr(note, "score", None)
        notes.append({"text": text, "score": score, "created_at": created_at})

    notes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return notes

# ---------------------
# STREAMLIT UI
# ---------------------
st.set_page_config(page_title="Audio Notatki", layout="centered")
st.title("🔐 Audio Notatki z OpenAI i Qdrant")

# OpenAI API Key
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

st.session_state["openai_api_key"] = st.text_input(
    "Twój OpenAI API Key",
    type="password",
    help="Klucz jest używany tylko w tej sesji i nigdzie nie jest zapisywany",
)

if not st.session_state["openai_api_key"]:
    st.warning("🔑 Podaj swój OpenAI API Key, aby korzystać z aplikacji")
    st.stop()

# Session state
for key in ["note_audio_bytes_md5", "note_audio_bytes", "note_text", "note_audio_text"]:
    if key not in st.session_state:
        st.session_state[key] = None if "bytes" in key else ""

assure_db_collection_exists()

add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])

# ---------------------
# Dodawanie notatki
# ---------------------
with add_tab:
    note_audio = audiorecorder("Nagraj notatkę", "Zatrzymaj nagrywanie")
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        if st.button("Transkrybuj audio"):
            with st.spinner("Transkrybuję audio..."):
                st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])

        if st.session_state["note_audio_text"]:
            st.session_state["note_text"] = st.text_area(
                "Edytuj notatkę",
                value=st.session_state["note_audio_text"]
            )

        if st.button("Zapisz notatkę", disabled=not st.session_state["note_text"]):
            add_note_to_db(st.session_state["note_text"])

# ---------------------
# Wyszukiwanie notatki
# ---------------------
with search_tab:
    query = st.text_input("Wyszukaj notatkę")
    if st.button("Szukaj") or query == "":
        notes = list_notes_from_db(query if query else None)
        if not notes:
            st.info("Brak notatek do wyświetlenia.")
        for note in notes:
            with st.container():
                st.markdown(note["text"])
                if note["score"] is not None:
                    st.markdown(f':violet[{note["score"]}]')
