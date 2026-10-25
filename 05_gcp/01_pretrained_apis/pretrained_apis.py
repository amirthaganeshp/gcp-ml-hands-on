"""Pre-trained APIs: the ones where you write no model at all.

Google Cloud sells machine learning at four levels of effort, and the most
valuable skill the exam tests is choosing between them rather than
implementing any of them. This folder is the first level: a REST call
against a model someone else trained, tuned and serves.

  1. The four levels, and the question that picks between them
  2. What each pre-trained API actually does, and its real limits
  3. Real client code for the five you are most likely to be asked about
  4. Cost and latency, which decide more architectures than accuracy does

NOTHING HERE CALLS GOOGLE CLOUD. Every cloud import sits inside a function,
so this script runs and teaches with no credentials, no project and no
billing. The code in those functions is real and deployable; it has not
been executed, and anywhere a response is shown it is labelled as
illustrative rather than captured.

Run:
    python 05_gcp/01_pretrained_apis/pretrained_apis.py

Concept reference: Foundations volume, Part D.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "00_setup"))


def banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


# ====================================================== 1. the four levels
def the_four_levels() -> None:
    rows = [
        ("Pre-trained API", "none", "none", "minutes",
         "a common task on common data"),
        ("AutoML", "your labels", "none", "hours",
         "your data, standard task, no ML staff"),
        ("BigQuery ML", "your table", "SQL only", "minutes",
         "the data is already in BigQuery"),
        ("Custom training", "your data", "you write it", "days",
         "a custom architecture or loss"),
    ]
    print(f"  {'level':<18} {'data':<12} {'model code':<14} {'to first result':<16} "
          f"use when")
    print("  " + "-" * 92)
    for a, b, c, d, e in rows:
        print(f"  {a:<18} {b:<12} {c:<14} {d:<16} {e}")

    print("\n  THE QUESTION THAT PICKS BETWEEN THEM is not 'which is most")
    print("  accurate' - it is 'how unusual is my problem'.")
    print("\n    Is it a task thousands of companies have?   -> pre-trained API")
    print("    Standard task, but MY data is the point?    -> AutoML")
    print("    Is the data already in BigQuery?            -> BigQuery ML")
    print("    Do I need a custom architecture or loss?    -> custom training")
    print("\n  Exam questions in this area almost always describe a team with")
    print("  no ML engineers and a tight deadline, and the correct answer is")
    print("  almost always the LEAST custom option that fits. A question that")
    print("  mentions 'no machine learning expertise' or 'as quickly as")
    print("  possible' is steering you away from custom training.")
    print("\n  The reverse trap exists too: if the question describes a novel")
    print("  architecture, a custom loss, or research, then AutoML is wrong")
    print("  no matter how convenient it sounds.")


# =============================================== 2. what each API does
def the_apis() -> None:
    apis = [
        ("Vision AI", "images",
         "labels, objects, OCR, faces, logos, safe-search, landmarks",
         "OCR on documents - use Document AI instead, it understands layout"),
        ("Document AI", "PDFs, scans",
         "layout-aware extraction, forms, tables, specialised parsers "
         "(invoice, receipt, ID)",
         "plain text with no structure - Vision OCR is cheaper"),
        ("Natural Language AI", "text",
         "entities, sentiment, syntax, content classification",
         "domain-specific sentiment - it is trained on general text"),
        ("Speech-to-Text", "audio",
         "transcription, diarisation, punctuation, word timings",
         "heavy jargon or accents without a custom phrase set"),
        ("Translation AI", "text",
         "100+ language pairs, glossary support, batch and adaptive",
         "legal or medical text where a mistranslation carries risk"),
    ]
    for name, inp, does, careful in apis:
        print(f"  {name}")
        print(f"      input     {inp}")
        print(f"      gives you {does}")
        print(f"      careful   {careful}\n")

    print("  Two distinctions the exam reliably tests:")
    print("\n    VISION OCR vs DOCUMENT AI. Vision returns text and where it")
    print("    sat on the page. Document AI understands that a page has")
    print("    fields, tables and a layout, and the specialised parsers")
    print("    return typed entities - invoice total, due date, supplier.")
    print("    'Extract line items from invoices' is Document AI every time.")
    print("\n    PRE-TRAINED vs AUTOML for the same modality. Vision AI knows")
    print("    'dog', 'car', 'tree'. It does not know your company's part")
    print("    numbers or defect classes. The moment the labels are YOURS,")
    print("    the answer moves to AutoML (now Vertex AI training) or custom.")


# ============================================ 3. the client code, for real
def vision_labels(image_path: str, max_results: int = 5):
    """Detect labels in an image. Requires google-cloud-vision and ADC.

    The import is inside the function deliberately - this module must import
    cleanly with no cloud libraries installed. That is also a decent pattern
    for real code: it keeps a heavy, credential-dependent dependency out of
    the import path of anything that merely wants to read this file.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as f:
        image = vision.Image(content=f.read())

    response = client.label_detection(image=image, max_results=max_results)
    if response.error.message:
        # the API reports errors in the BODY, not as an exception - a very
        # common source of silent failure
        raise RuntimeError(response.error.message)
    return [(l.description, l.score) for l in response.label_annotations]


def document_ai_extract(project: str, location: str, processor_id: str,
                        pdf_path: str):
    """Parse a document with a Document AI processor.

    Note the processor is a RESOURCE YOU CREATE in the console first, and it
    has a type (invoice parser, form parser, OCR). The processor id here is
    not optional configuration - it decides what you get back.
    """
    from google.cloud import documentai

    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project, location, processor_id)
    with open(pdf_path, "rb") as f:
        raw = documentai.RawDocument(content=f.read(),
                                     mime_type="application/pdf")
    result = client.process_document(
        request=documentai.ProcessRequest(name=name, raw_document=raw))
    doc = result.document
    return {
        "text": doc.text[:500],
        "entities": [(e.type_, e.mention_text, e.confidence)
                     for e in doc.entities],
        "pages": len(doc.pages),
    }


def analyse_sentiment(text: str):
    """Document-level sentiment: score in [-1, 1], magnitude in [0, inf).

    The pair matters and is regularly misread. SCORE is direction.
    MAGNITUDE is strength, and it accumulates with length - so a long,
    strongly mixed review has a score near 0 and a LARGE magnitude, while a
    short neutral one has score near 0 and a small magnitude. Reading score
    alone makes those two identical.
    """
    from google.cloud import language_v1

    client = language_v1.LanguageServiceClient()
    doc = language_v1.Document(content=text,
                               type_=language_v1.Document.Type.PLAIN_TEXT)
    s = client.analyze_sentiment(request={"document": doc}).document_sentiment
    return {"score": s.score, "magnitude": s.magnitude}


def transcribe(gcs_uri: str, language: str = "en-US"):
    """Speech-to-Text on a file in Cloud Storage.

    Anything over about a minute must use long_running_recognize and a GCS
    URI - the synchronous API caps out. That asymmetry is exam material.
    """
    from google.cloud import speech

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language,
        enable_automatic_punctuation=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
    )
    op = client.long_running_recognize(
        config=config, audio=speech.RecognitionAudio(uri=gcs_uri))
    response = op.result(timeout=600)
    return " ".join(r.alternatives[0].transcript for r in response.results)


def translate_text(project: str, text: str, target: str = "es"):
    """Translation v3. Note it is project-scoped, unlike v2 - the parent
    resource path is required, and glossaries and models attach to it.
    """
    from google.cloud import translate

    client = translate.TranslationServiceClient()
    parent = f"projects/{project}/locations/global"
    response = client.translate_text(
        request={"parent": parent, "contents": [text],
                 "mime_type": "text/plain", "target_language_code": target})
    return response.translations[0].translated_text


def show_client_code() -> None:
    print("  Five functions are defined above, ready to run once you have")
    print("  credentials. They have NOT been executed - see the note at the")
    print("  top of this file. Read them for the shapes, not the outputs.\n")
    for fn in (vision_labels, document_ai_extract, analyse_sentiment,
               transcribe, translate_text):
        first = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"    {fn.__name__:<22} {first}")

    print("\n  Three things those functions have in common, and all three are")
    print("  the sort of detail that only shows up when you actually run one:")
    print("\n    AUTHENTICATION IS IMPLICIT. None of them takes a credential.")
    print("    They use Application Default Credentials - a service account")
    print("    on GCP, or `gcloud auth application-default login` locally.")
    print("    Passing a key file explicitly is the discouraged path.")
    print("\n    ERRORS COME BACK IN THE BODY. Vision in particular returns")
    print("    response.error.message rather than raising, so code that does")
    print("    not check it fails silently and returns nothing.")
    print("\n    LONG JOBS ARE OPERATIONS, NOT CALLS. Speech over ~1 minute")
    print("    returns a long-running operation you poll, and the audio has")
    print("    to be in Cloud Storage rather than the request body.")


# ================================================ 4. cost and latency
def cost_and_latency() -> None:
    print("  Rough orders of magnitude - check current pricing before")
    print("  quoting any of these, they change and they vary by region:\n")
    rows = [
        ("Vision label detection", "~$1.50 / 1,000 images", "~300 ms"),
        ("Document AI form parser", "~$30-65 / 1,000 pages", "~5-10 s/page"),
        ("Natural Language sentiment", "~$1 / 1,000 requests", "~200 ms"),
        ("Speech-to-Text", "~$0.02 / minute", "async, minutes"),
        ("Translation", "~$20 / million characters", "~200 ms"),
    ]
    print(f"  {'service':<28} {'rough cost':<26} typical latency")
    for a, b, c in rows:
        print(f"  {a:<28} {b:<26} {c}")

    print("\n  These numbers decide more architectures than accuracy does, and")
    print("  the exam knows it. Two patterns worth recognising:")
    print("\n    BATCH BEATS ONLINE ON COST. If the answer is not needed in the")
    print("    request path, batch endpoints are materially cheaper and you")
    print("    stop paying for an always-warm endpoint.")
    print("\n    DOCUMENT AI IS EXPENSIVE PER PAGE. At 30-65 dollars per")
    print("    thousand pages, a million-page backlog is a real budget line.")
    print("    A question that mentions high volume and simple text is")
    print("    steering you to Vision OCR, which is roughly an order of")
    print("    magnitude cheaper.")
    print("\n  And the architectural point underneath all of it: a pre-trained")
    print("  API has NO TRAINING COST AND NO SERVING COST AT REST. You pay per")
    print("  call and nothing when idle. A custom model on a Vertex endpoint")
    print("  bills for the node whether or not anyone calls it - which is why")
    print("  'low, intermittent traffic' in a question usually argues against")
    print("  a deployed endpoint.")


def main() -> None:
    print("=" * 72)
    print("  PRE-TRAINED APIs")
    print("=" * 72)
    print("\n  Write-only: no cloud calls, no credentials, no billing.")
    print("  The client code is real; it has not been run.")

    banner("1. The four levels of effort")
    the_four_levels()

    banner("2. What each API does, and where it stops")
    the_apis()

    banner("3. The client code")
    show_client_code()

    banner("4. Cost and latency")
    cost_and_latency()

    banner("What to take away")
    print("  - Pick the LEAST CUSTOM option that fits. The exam rewards this")
    print("    heavily, and 'no ML expertise' or 'as fast as possible' in a")
    print("    question is a direct signal.")
    print("  - Pre-trained APIs know GENERAL categories. The moment the labels")
    print("    are yours, the answer moves to AutoML or custom training.")
    print("  - Vision OCR extracts text; DOCUMENT AI understands layout and")
    print("    returns typed entities. 'Invoice line items' is Document AI.")
    print("  - Sentiment has SCORE (direction) and MAGNITUDE (strength). A")
    print("    long mixed review and a short neutral one both score ~0 and are")
    print("    told apart only by magnitude.")
    print("  - Authentication is implicit via Application Default Credentials.")
    print("    Explicit key files are the discouraged path.")
    print("  - Long audio is a LONG-RUNNING OPERATION reading from GCS, not a")
    print("    synchronous call.")
    print("  - Pre-trained APIs cost nothing at rest. A deployed endpoint")
    print("    bills whether or not it is called - which is the usual argument")
    print("    against one for low, intermittent traffic.")
    print("=" * 72)


if __name__ == "__main__":
    main()
