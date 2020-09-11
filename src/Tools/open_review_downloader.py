# This file uses python3.8 and depends on

# example usage
# python3 src/Tools/open_review_downloader.py \
#   --username="USEERNAME_HERE"\
#   --password="PASSWORD_HERE"\
#   --base_path="PATH_TO_OUTPUT_DIR_HERE"

# Requirements, put this in a requirements.txt file and run pip3 install -r requirements.txt
#  certifi==2020.6.20
#  chardet==3.0.4
#  cycler==0.10.0
#  Deprecated==1.2.10
#  future==0.18.2
#  idna==2.10
#  kiwisolver==1.2.0
#  matplotlib==3.3.1
#  numpy==1.19.1
#  openreview-py==1.0.17
#  pandas==1.1.2
#  pdf2image==1.14.0
#  Pillow==7.2.0
#  pycryptodome==3.9.8
#  pylatexenc==2.7
#  pyparsing==2.4.7
#  PyPDF2==1.26.0
#  python-dateutil==2.8.1
#  pytz==2020.1
#  requests==2.24.0
#  six==1.15.0
#  tld==0.10
#  tqdm==4.48.2
#  urllib3==1.25.10
#  wrapt==1.12.1


import openreview
import logging
from PIL import Image
import numpy as np
import time
import PyPDF2
import os
from PyPDF2.utils import PdfReadError
import pdf2image
import io
from dataclasses import dataclass
from typing import Iterator, List, Optional
import pathlib
import argparse
import pandas as pd


@dataclass
class ScrapeURLs:
    submission_notes: str
    decision_notes: str


@dataclass
class Paper:
    id: str
    authors: List[str]
    title: str
    abstract: str
    pdf: bytes
    accepted: bool


DEFAULT_SOURCES = [
    ScrapeURLs(submission_notes="MIDL.io/2019/Conference/-/Full_Submission",
               decision_notes="MIDL.io/2019/Conference/-/Paper.*/Decision")

]

LOGGER_NAME = "OpenReviewScraper"


class OpenReviewScraper:

    def __init__(self, username: str, password: str):
        """Instantiate OpenReview Scraper.

        Args:
            username: OpenReview username e.g email address used to create account
            password: OpenReview password used to create account

        Raises:
            OpenReviewException: If the username or password is invalid
        """
        self.logging = logging.getLogger(LOGGER_NAME)

        self.logging.info(f"Trying to create client with - user {username} - password {password}")
        self.client = openreview.Client(baseurl='https://api.openreview.net',
                                        username=username,
                                        password=password)

        self.logging.info(f"Successfully connected")

    def __call__(self, sources: List[ScrapeURLs] = None) -> Iterator[Paper]:
        """Scrapes openReview for papers.

        The pdf field in the returned papers is raw PDF files and
        will need to be post-processed to extract the cover image.

        Args:
            sources: List of sources to scrape - see DEFAULT_SOURCES for example

        Returns:
            An iterator of type Paper
        """
        if sources is None:
            sources = DEFAULT_SOURCES

        self.logging.info(f"Using sources: {sources}")

        for source in sources:
            self.logging.info(f"Scraping {source}")
            decision_notes = {note.forum: note.content["decision"]
                              for note in openreview.tools.iterget_notes(self.client,
                                                                         invitation=source.decision_notes)}

            for note in openreview.tools.iterget_notes(self.client, invitation=source.submission_notes):
                try:
                    yield Paper(id=note.id,
                                authors=note.content["authors"],
                                title=note.content["title"],
                                abstract=note.content["abstract"],
                                pdf=self.client.get_pdf(note.id),
                                accepted=decision_notes[note.id] == "Accept")
                    self.logging.info(f"Downloaded {note.content['title']}")
                except openreview.openreview.OpenReviewException as e:
                    self.logging.warning(f"Failed to create paper from {note.id}, reason: {str(e)}")
                except KeyError as e:
                    self.logging.warning(f"Failed to create paper from {note.id}, missing field: {str(e)}")


@dataclass
class ImagePaper:
    id: str
    authors: List[str]
    title: str
    abstract: str
    image: np.ndarray
    accepted: bool


def convert_Paper_to_ImagePaper(paper: Paper) -> Optional[ImagePaper]:
    """Convert the paper into a ImagePaper to be used by our vision model.

    This will extract the first page of the pdf and convert it into an image, all in memory.

    Args:
        paper: The paper dataclass containing the PDF information
    Returns:
        Optional[ImagePaper] - a dataclass containing a numpy array of width x height x channels --  or None
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"Attempting to convert PDF {paper.title}")

    try:
        # A fake input file
        pdf_stream = io.BytesIO(paper.pdf)

        pdf = PyPDF2.PdfFileReader(pdf_stream)

        # Get first page
        first_page = pdf.getPage(0)

        # Now get raw pdf data
        first_page_bytes = io.BytesIO()

        pdf_writer = PyPDF2.PdfFileWriter()
        pdf_writer.addPage(first_page)
        pdf_writer.write(first_page_bytes)

        # pdf2image will create on image per page, but since we only have one we get the first one
        pil_image = pdf2image.convert_from_bytes(first_page_bytes.getvalue())[0]

        image_data = np.array(pil_image)

        return ImagePaper(id=paper.id,
                          authors=paper.authors,
                          title=paper.title,
                          abstract=paper.abstract,
                          image=image_data,
                          accepted=paper.accepted)
    except PdfReadError as e:
        logger.warning(f"Unable to read {paper.title} reason: {e} -- discarding it")


def ImagePapers_to_dataset(ipapers: List[ImagePaper], base_path: str, image_infix: str = "images") -> None:
    """Takes a list of ImagePapers and constructs a csv with meta information and stores images as png files.

    Args:
        ipapers: List of ImagePapers
        base_path: Path to construct dataset -- meta data will be saved in base_path/meta.csv and images
                   in base_path/images/...
        image_infix: Name of subpath in which image files is stored
    """
    logger = logging.getLogger(LOGGER_NAME)

    base_path = pathlib.Path(base_path)
    if not base_path.is_dir():
        logger.warning(f"{base_path} is not a valid directory")
        logger.warning(f"Creating {base_path}")

        # Assume it works.. if someone can be bothered to add fail-safes, do so
        os.mkdir(base_path)

    image_path = base_path / image_infix
    if not image_path.is_dir():
        logger.warning(f"{image_path} is not a valid directory")
        logger.warning(f"Creating {image_path}")

        # Assume it works.. if someone can be bothered to add fail-safes, do so
        os.mkdir(image_path)

    meta_features = {
        "id": [],
        "authors": [],
        "title": [],
        "abstract": [],
        "accepted": [],

        "image_path": [],
    }

    for paper in ipapers:
        # Add all meta features
        meta_features["id"].append(paper.id)
        meta_features["authors"].append("-".join(paper.authors))
        meta_features["title"].append(paper.title)
        meta_features["abstract"].append(paper.abstract)
        meta_features["accepted"].append(paper.accepted)

        relative_image_path = f"{image_infix}/{paper.id}.png"
        meta_features["image_path"].append(relative_image_path)

        absolute_image_path = base_path / relative_image_path
        # Write image to disk

        logger.info(f"Storing {paper.title} at -- {absolute_image_path}")
        with open(absolute_image_path, "wb") as image_file:
            Image.fromarray(paper.image).save(image_file)

    pd.DataFrame(meta_features).to_csv(base_path / "meta.csv", index=False)


def make_dataset(sources: List[ScrapeURLs],
                 dataset_base_path: str,
                 open_review_username: str,
                 open_review_password: str):
    """Constructs a dataset using the OpenReview API

    The dataset will be a csv written to 'dataset_base_path/meta.csv' containing the fields
        id: The paper identifier on OpenReview
        authors: The authors of the paper separated with "-"
        title: The title of the paper
        abstract: The abstract of the paper
        accepted: If it was accepted or not, True indicates accepted
        image_path: A path relative to 'dataset_base_path' to an image of the paper

    Args:
        sources: The conferences to scrape -- see "DEFAULT_SOURCES" at the top of this file for an example
        dataset_base_path: An absolute path to the base of the dataset
        open_review_username: Username to the OpenReview website (e.g email used to sign up)
        open_review_password: Password to the OpenReview website

    Raises:
        #TODO(jonasrsv, ...): In what way does this crash? :)

    Returns:
        None, this function constructs the dataset at 'dataset_base_path'
    """
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("Fetching papers")
    papers = [paper for paper in OpenReviewScraper(username=open_review_username, password=open_review_password)(
        sources=sources
    )]

    logger.info("Extracting Images")
    ipapers = [convert_Paper_to_ImagePaper(paper) for paper in papers]

    logger.info("Building dataset")
    ImagePapers_to_dataset(ipapers=ipapers, base_path=dataset_base_path, image_infix="images")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, help="Username to OpenReview", required=True)
    parser.add_argument("--password", type=str, help="Password to OpenReview", required=True)
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)

    args = parser.parse_args()

    logger.info(f"Making dataset at {args.base_path} with default sources")

    timestamp = time.time()
    make_dataset(sources=DEFAULT_SOURCES, dataset_base_path=args.base_path,
                 open_review_username=args.username, open_review_password=args.password)

    logger.info(f"Construction of dataset took {time.time() - timestamp} seconds")


