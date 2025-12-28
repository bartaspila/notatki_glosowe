from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo


EMBEDDING_MODEL = "text-embedding-3-large" 

EMBEDDING_DIM = 3072

AUDIO_TRANSCRIBE_MODEL = "whisper-1"

QDRANT_COLLECTION_NAME = "notes"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])


def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"

    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
    )

    return transcript.text



def debug_collections():
    st.write(get_qdrant_client().get_collections())

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )

    
def assure_db_collection_exists():
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]  # <- tu jest lista CollectionInfo
    if QDRANT_COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        print("Utworzono kolekcję")
    else:
        print("Kolekcja już istnieje")

def get_embedding(text):
    text = text[:8000]
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        # dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    qdrant_client.upsert(
    collection_name=QDRANT_COLLECTION_NAME,
    points=[
        PointStruct(
            # id=points_count.count + 1,
            id=str(uuid.uuid4()),
            vector=get_embedding(text=note_text),
            payload={
                "text": note_text,
            },
        )
    ]
    )


def list_notes_from_db(query=None):
    client = get_qdrant_client()
    result = []

    if not query:
        # Pobierz 10 ostatnich punktów
        response = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=10,
        )
        points = response.points  # w nowszej wersji .points istnieje
    else:
        # Szukaj podobnych punktów semantycznie
        points = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(query),
            limit=10,
        )  # zwraca od razu List[PointStruct]

    for note in points:
        text = note.payload.get("text") if note.payload else ""
        result.append({
            "text": text,
            "score": getattr(note, "score", None),
        })

    return result




#
# MAIN
#
st.set_page_config(page_title="Audio Notatki", layout="centered")

# 🔐 Poproś użytkownika o własny OpenAI API Key
st.title("🔐 Dostęp do OpenAI")

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

# test do wyszukania notatek
# st.write(list_notes_from_db())


# Session state initialization
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

assure_db_collection_exists()
st.title("Audio Notatki")

add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])
with add_tab:
    note_audio = audiorecorder(
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )
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
            st.session_state["note_text"] = st.text_area("Edytuj notatkę", value=st.session_state["note_audio_text"])

        if st.button("Zapisz notatkę", disabled=not st.session_state["note_text"]):
            if st.session_state["note_text"]:

                # Ograniczenie długości notatki
                MAX_CHARS = 8000
                note_text = st.session_state["note_text"][:MAX_CHARS]
                # Zapisujemy ograniczoną notatkę
                add_note_to_db(note_text=note_text)
                # add_note_to_db(note_text=st.session_state["note_text"])
                st.success("Notatka zapisana 🎉")
                

with search_tab:
    query = st.text_input("Wyszukaj notatkę")
    if st.button("Szukaj"):
        for note in list_notes_from_db(query):
            with st.container(border=True):
                st.markdown(note["text"])
                if note["score"] is not None:
                    st.markdown(f':violet[{note["score"]}]')
