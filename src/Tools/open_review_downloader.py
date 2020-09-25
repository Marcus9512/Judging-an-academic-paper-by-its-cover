# This file uses python3.8 and depends on

# example usage
# python3 src/Tools/open_review_downloader.py \
#   --username="USEERNAME_HERE"\
#   --password="PASSWORD_HERE"\
#   --base_path="PATH_TO_OUTPUT_DIR_HERE"\
#   --num_threads=10\

import openreview
import logging
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ConnectionError
import sys
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
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


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
    # TODO(...): Need to work a bit more with the 'accept' extraction on the ICLR thingies
    # ScrapeURLs(submission_notes="ICLR.cc/2019/Conference/-/Blind_Submission",
    #           decision_notes="ICLR.cc/2019/Conference/-/Paper.*/Official_Review"),

    # These ones works as intended
    ScrapeURLs(submission_notes="ICLR.cc/2020/Conference/-/Blind_Submission",
               decision_notes="ICLR.cc/2020/Conference/Paper.*/-/Decision"),
    ScrapeURLs(submission_notes="MIDL.io/2020/Conference/-/Blind_Submission",
               decision_notes="MIDL.io/2020/Conference/Paper.*/-/Decision"),
    ScrapeURLs(submission_notes="MIDL.io/2019/Conference/-/Full_Submission",
               decision_notes="MIDL.io/2019/Conference/-/Paper.*/Decision"),
    ScrapeURLs(submission_notes="MIDL.amsterdam/2018/Conference/-/Submission",
               decision_notes="MIDL.amsterdam/2018/Conference/-/Paper.*/Acceptance_Decision"),

    #Seems to work , gives 917 rejected, 502 accepted
    ScrapeURLs(submission_notes="ICLR.cc/2019/Conference/-/Blind_Submission",
           decision_notes="ICLR.cc/2019/Conference/-/Paper.*/Meta_Review"),

    #Seems to work, gives 598 rejected, 337 accepted
    ScrapeURLs(submission_notes="ICLR.cc/2018/Conference/-/Blind_Submission",
           decision_notes="ICLR.cc/2018/Conference/-/Acceptance_Decision"),


    ##### WORKSHOPS
    ScrapeURLs(submission_notes="ICLR.cc/2018/Workshop/-/Submission",
           decision_notes="ICLR.cc/2018/Workshop/-/Acceptance_Decision"),

    ScrapeURLs(submission_notes="ICLR.cc/2019/Workshop/.*/-/Blind_Submission",
            decision_notes="ICLR.cc/2019/Workshop/.*/-/Paper.*/Decision"),

    ScrapeURLs(submission_notes="ICLR.cc/2020/Workshop/.*/-/Blind_Submission",
            decision_notes="ICLR.cc/2020/Workshop/.*/Paper.*/-/Decision"),
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

    def __is_accepted(self, note):
        """This applies heuristics to see if a paper is accepted."""

        # If the paper is from MIDL it is nice, because they have an decision field
        # We can simply look in that field and extract the information
        # Apparently for some years the ICLR also have this decision field, which is nice
        if "decision" in note.content:
            # TODO(...): Improve this heuristic
            return (("Accept" in note.content["decision"])  # If it contains Accept it is accept
                    or ("Poster" == note.content["decision"])  # In MIDL 2018 decisions are (Reject | Oral | Poster)
                    or ("Oral" == note.content["decision"])  # In MIDL 2018 decisions are (Reject | Oral | Poster)
                    )
        # If it is from ICLR and it is not a year with decision field its a bit more tedious --
        # Instead in their 'Official Reviews' there is a field called 'rating'
        # The first character of the rating is a number 0-9 if it is larger than 5 it appears to be accepted.

        #Works for ICLR 2019
        if "recommendation" in note.content:
            return "Accept" in note.content["recommendation"]

        # Commented this out, not sure if this is needed anymore
        # if 'rating' in note.content:
        #     # TODO(..): Work on this heuristic
        #     if int(note.content['rating'][0]) > 5:
        #         return True
        #     return False

        raise Exception("Cannot determine if a paper is accepted or not.")

    def __call__(self, sources: List[ScrapeURLs] = None, num_threads: int = 1) -> Iterator[Paper]:
        """Scrapes openReview for papers.

        The pdf field in the returned papers is raw PDF files and
        will need to be post-processed to extract the cover image.

        Args:
            sources: List of sources to scrape - see DEFAULT_SOURCES for example
            num_threads: How many threads to use while downloading PDF's

        Returns:
            An iterator of type Paper
        """
        if sources is None:
            sources = DEFAULT_SOURCES

        self.logging.info(f"Using sources: {sources}")

        for source in sources:
            self.logging.info(f"Scraping {source}")

            self.logging.info(f"Downloading decision notes...")
            decision_notes = {note.forum: self.__is_accepted(note)
                              for note in openreview.tools.iterget_notes(self.client,
                                                                         invitation=source.decision_notes)}
            number_decisions = len(decision_notes)
            self.logging.info(f"Downloaded {number_decisions} decision notes")
            self.logging.info(f"Downloading submission notes...")
            submission_notes = list(openreview.tools.iterget_notes(self.client, invitation=source.submission_notes))
            number_submission = len(submission_notes)
            self.logging.info(f"Downloaded {number_submission} submission notes")

            num_accepted, num_rejected = 0, 0

            def downloader_fn(note: openreview.Note):
                # Ok with python threads
                nonlocal num_rejected, num_accepted

                paper = None
                try:
                    paper = Paper(id=note.id,
                                  authors=note.content["authors"],
                                  title=note.content["title"],
                                  abstract=note.content["abstract"],
                                  pdf=self.client.get_pdf(note.id),
                                  accepted=decision_notes[note.id])

                    # This is okay when using the normal thread-pool since all threads
                    # run in the same process space & thanks to python's GIL we get no race-errors.
                    num_accepted += paper.accepted
                    num_rejected += (1 - paper.accepted)

                    return paper
                except openreview.openreview.OpenReviewException as e:
                    self.logging.warning(f"Failed to create paper from {note.id}, reason: {str(e)}")
                except KeyError as e:
                    self.logging.warning(f"Failed to create paper from {note.id}, missing field: {str(e)}")
                except MaxRetryError as e:
                    self.logging.warning(f"Failed to create paper {note.content['title']} -- Reason: {e}")
                except ConnectionError as e:
                    self.logging.warning(f"Failed to create paper {note.content['title']} -- Reason: {e}")
                except TimeoutError as e:
                    self.logging.warning(f"Failed to create paper {note.content['title']} -- Reason: {e}")
                except Exception as e:
                    self.logging.warning(
                        f"Unknown Exception: Failed to create paper {note.content['title']} -- Reason: {e}")

                return paper

            with ThreadPool(processes=num_threads) as pool:
                with tqdm(total=number_submission) as progress_bar:
                    for note in pool.imap_unordered(downloader_fn, submission_notes):
                        # update progress bar
                        progress_bar.update()

                        yield note

            self.logging.info(f"Accepted: {num_accepted} -- Rejected: {num_rejected}")


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
    try:
        # A fake input file
        pdf_stream = io.BytesIO(paper.pdf)

        pdf = PyPDF2.PdfFileReader(pdf_stream)

        # Get first page that is a valid front-page
        def _is_valid_frontpage(page: PyPDF2.pdf.PageObject) -> bool:
            # TODO(...) A better heuristic, 50 is completely arbitrary at the moment
            return len(page.extractText()) > 50

        for page_index in range(pdf.numPages):
            front_page = pdf.getPage(page_index)

            if _is_valid_frontpage(front_page):
                break
        else:
            raise ValueError("No page is a valid front-page")

        # Now get raw pdf data
        front_page_bytes = io.BytesIO()

        pdf_writer = PyPDF2.PdfFileWriter()
        pdf_writer.addPage(front_page)
        pdf_writer.write(front_page_bytes)

        # pdf2image will create on image per page, but since we only have one we get the first one
        pil_image = pdf2image.convert_from_bytes(front_page_bytes.getvalue())[0]

        image_data = np.array(pil_image)

        return ImagePaper(id=paper.id,
                          authors=paper.authors,
                          title=paper.title,
                          abstract=paper.abstract,
                          image=image_data,
                          accepted=paper.accepted)
    except PdfReadError as e:
        logger.warning(f"PdfReadError: Unable to read {paper.title} reason: {e} -- discarding it")
    except ValueError as e:
        logger.warning(f"ValueError: Paper {paper.title} was not converted to ImagePaper, Reason: {e}")
    except TypeError as e:
        logger.warning(f"TypeError: Paper {paper.title} was not converted to ImagePaper, Reason: {e}")
    except Exception as e:
        logger.warning(f"Unknown Exception: {e} -- Dropping Paper {paper.title}")


def ImagePapers_to_dataset(ipapers: List[ImagePaper], base_path: str, image_infix: str = "images") -> None:
    """Takes a list of ImagePapers and constructs a csv with meta information and stores images as png files.

    This will lazily download and convert the papers to avoid having it all in memory at once.

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

    logger.info(f"Storing papers at -- {base_path}")
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

        with open(absolute_image_path, "wb") as image_file:
            Image.fromarray(paper.image).save(image_file)

    pd.DataFrame(meta_features).to_csv(base_path / "meta.csv", index=False)


def make_dataset(sources: List[ScrapeURLs],
                 dataset_base_path: str,
                 open_review_username: str,
                 open_review_password: str,
                 num_threads: int):
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
        num_threads: The number of threads to use to download PDFs

    Raises:
        #TODO(jonasrsv, ...): In what way does this crash? :)

    Returns:
        None, this function constructs the dataset at 'dataset_base_path'
    """
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("Starting Download & Conversion of papers")
    # Lazy downloading papers, Downloading all at once won't fit in memory.. just one ICML + MIDL takes more than
    # 30GB memory
    papers = (paper for paper in OpenReviewScraper(username=open_review_username, password=open_review_password)(
        sources=sources, num_threads=num_threads
    ) if paper is not None)

    # Lazy converting papers
    ipapers = (paper for paper in map(convert_Paper_to_ImagePaper, papers) if paper is not None)

    ImagePapers_to_dataset(ipapers=ipapers, base_path=dataset_base_path, image_infix="images")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, help="Username to OpenReview", required=True)
    parser.add_argument("--password", type=str, help="Password to OpenReview", required=True)
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    parser.add_argument("--num_threads", type=int, help="Number of threads to run download in", default=5)

    args = parser.parse_args()

    logger.info(f"Making dataset at {args.base_path} with default sources")

    timestamp = time.time()
    make_dataset(sources=DEFAULT_SOURCES, dataset_base_path=args.base_path,
                 open_review_username=args.username, open_review_password=args.password,
                 num_threads=args.num_threads)

    logger.info(f"Construction of dataset took {time.time() - timestamp} seconds")
