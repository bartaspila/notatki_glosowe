from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
# from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import uuid
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo

# env = dotenv_values(".env")

EMBEDDING_MODEL = "text-embedding-3-large" # 

EMBEDDING_DIM = 3072

AUDIO_TRANSCRIBE_MODEL = "whisper-1"

QDRANT_COLLECTION_NAME = "notes"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

# def transcribe_audio(audio_bytes):
#     openai_client = get_openai_client()
#     audio_file = BytesIO(audio_bytes)
#     audio_file.name = "audio.mp3"
    # transcript = openai_client.audio.transcriptions.create(
    #     file=audio_file,
    #     model=AUDIO_TRANSCRIBE_MODEL,
    #    response_format="verbose_json",
    #)

    # return transcript.text
def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"

    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
    )

    return transcript.text

#
# DB
#
# @st.cache_resource
# def get_qdrant_client():
#     return QdrantClient( # path=":memory:")
# #qdrant_client = QdrantClient(
#     url="https://4be186c6-f073-4d00-a5b9-5e3f2ba313dd.europe-west3-0.gcp.cloud.qdrant.io:6333", 
#     api_key=env["QDRANT_API_KEY"],
# )
# @st.cache_resource
# def get_qdrant_client():
#     return QdrantClient(
#         url="https://4be186c6-f073-4d00-a5b9-5e3f2ba313dd.europe-west3-0.gcp.cloud.qdrant.io:6333",
#         api_key=env.get("QDRANT_API_KEY"),
#     )
#     if not env.get("QDRANT_API_KEY"):
#         st.error("Brak QDRANT_API_KEY w .env")
#         st.stop()
# @st.cache_resource
# def get_qdrant_client():
#     if not env.get("QDRANT_API_KEY"):
#         st.error("Brak QDRANT_API_KEY w .env")
#         st.stop()

#     return QdrantClient(
#         url=env["QDRANT_URL"],
#         api_key=env["QDRANT_API_KEY"],
#     )

# print(get_qdrant_client.get_collections()) # type: ignore
def debug_collections():
    st.write(get_qdrant_client().get_collections())

# def assure_db_collection_exists():
#     qdrant_client = get_qdrant_client()
#     if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
#         print("Tworzę kolekcję")
#         qdrant_client.create_collection(
#             collection_name=QDRANT_COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=EMBEDDING_DIM,
#                 distance=Distance.COSINE,
#             ),
#         )
# def assure_db_collection_exists():
#     client = get_qdrant_client()
#     try:
#         client.get_collection(QDRANT_COLLECTION_NAME)
#     except UnexpectedResponse:
#         client.create_collection(
#             collection_name=QDRANT_COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=EMBEDDING_DIM,
#                 distance=Distance.COSINE,
#             ),
#         )

#     else:
#         print("Kolekcja już istnieje")
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        )
    


# def assure_db_collection_exists():
#     client = get_qdrant_client()
#     try:
#         client.get_collection(QDRANT_COLLECTION_NAME)
#         print("Kolekcja już istnieje")
#     except UnexpectedResponse:
#         client.create_collection(
#             collection_name=QDRANT_COLLECTION_NAME,
#             vectors_config=VectorParams(
#                 size=EMBEDDING_DIM,
#                 distance=Distance.COSINE,
#             ),
#         )
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
    # points_count = qdrant_client.count(
    #     collection_name=QDRANT_COLLECTION_NAME,
    #     exact=True,
    # )
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

# def list_notes_from_db(query=None):
#     qdrant_client = get_qdrant_client()
#     if not query:
#         # notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
#         points, _ = qdrant_client.scroll(
#             collection_name=QDRANT_COLLECTION_NAME,
#             limit=10
#         )
#     notes = points # type: ignore

#     result = []
#     for note in notes:
#         result.append({
#             "text": note.payload["text"], # type: ignore
#             "score": None,
#         })

#    # return result

#     else:
#         notes = qdrant_client.search(
#             collection_name=QDRANT_COLLECTION_NAME,
#             query_vector=get_embedding(text=query),
#             limit=10,
#         )
#         result = []
#         for note in notes:
#             result.append({
#                 "text": note.payload["text"], # type: ignore
#                 "score": note.score,
#             })

#         return result
def list_notes_from_db(query=None):
    client = get_qdrant_client()
    result = []

    if not query:
        response = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=10,
        )
        points = getattr(response, "points", [])  # <- scroll ma .points

        for note in points:
            text = note.payload.get("text") if note.payload else ""
            result.append({
                "text": text,
                "score": None,
            })

    else:
        # search zwraca od razu listę punktów
        notes = client.search( # type: ignore
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(query),
            limit=10,
        )

        for note in notes:
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


# OpenAI API key protection
# if not st.session_state.get("openai_api_key"):
#     if "OPENAI_API_KEY" in env:
#         st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

#     else:
#         st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
#         st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
#         if st.session_state["openai_api_key"]:
#             st.rerun()

# if not st.session_state.get("openai_api_key"):
#     st.stop()

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
                st.toast("Notatka zapisana", icon="🎉")

with search_tab:
    query = st.text_input("Wyszukaj notatkę")
    if st.button("Szukaj"):
        for note in list_notes_from_db(query):
            with st.container(border=True):
                st.markdown(note["text"])
                if note["score"] is not None:
                    st.markdown(f':violet[{note["score"]}]')
