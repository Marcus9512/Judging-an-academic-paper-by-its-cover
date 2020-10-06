# example usage
# python3 src/Tools/open_review_downloader.py \
#   --username="USERNAME_HERE"\
#   --password="PASSWORD_HERE"\
#   --base_path=PATH_TO_DATA_DIRECTORY_HERE\
#   --num_pages=8\
#   --image_width=256\
#   --image_height=256\
#   --num_threads=20

import openreview
import logging
import re
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ConnectionError
from PIL import Image, ImageOps
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
    conference: str
    year: int


DEFAULT_SOURCES = [
    # These ones works as intended
    # 106 Accepted 33 Rejected
    ScrapeURLs(submission_notes="MIDL.io/2020/Conference/-/Blind_Submission",
               decision_notes="MIDL.io/2020/Conference/Paper.*/-/Decision"),

    # 47 Accepted 13 Rejected
    ScrapeURLs(submission_notes="MIDL.io/2019/Conference/-/Full_Submission",
               decision_notes="MIDL.io/2019/Conference/-/Paper.*/Decision"),

    # 47 Accepted 36 Rejected
    ScrapeURLs(submission_notes="MIDL.amsterdam/2018/Conference/-/Submission",
               decision_notes=
               "MIDL.amsterdam/2018/Conference/-/Paper.*/Acceptance_Decision"),

    # Seems to work , gives 917 rejected, 502 accepted
    ScrapeURLs(submission_notes="ICLR.cc/2019/Conference/-/Blind_Submission",
               decision_notes="ICLR.cc/2019/Conference/-/Paper.*/Meta_Review"),

    # #Seems to work, gives 598 rejected, 337 accepted
    ScrapeURLs(submission_notes="ICLR.cc/2018/Conference/-/Blind_Submission",
               decision_notes="ICLR.cc/2018/Conference/-/Acceptance_Decision"),

    # 687 Accepted -- 1526 Rejected
    ScrapeURLs(submission_notes="ICLR.cc/2020/Conference/-/Blind_Submission",
               decision_notes="ICLR.cc/2020/Conference/Paper.*/-/Decision"),

    ##### WORKSHOPS
    # ScrapeURLs(submission_notes="ICLR.cc/2018/Workshop/-/Submission",
    #        decision_notes="ICLR.cc/2018/Workshop/-/Acceptance_Decision"),

    # ScrapeURLs(submission_notes="ICLR.cc/2019/Workshop/.*/-/Blind_Submission",
    #         decision_notes="ICLR.cc/2019/Workshop/.*/-/Paper.*/Decision"),

    # ScrapeURLs(submission_notes="ICLR.cc/2020/Workshop/.*/-/Blind_Submission",
    #         decision_notes="ICLR.cc/2020/Workshop/.*/Paper.*/-/Decision"),
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

        self.logging.info(
            f"Trying to create client with - user {username} - password {password}"
        )
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
            return (("Accept" in note.content["decision"]
                     )  # If it contains Accept it is accept
                    or
                    ("Poster" == note.content["decision"]
                     )  # In MIDL 2018 decisions are (Reject | Oral | Poster)
                    or
                    ("Oral" == note.content["decision"]
                     )  # In MIDL 2018 decisions are (Reject | Oral | Poster)
                    )
        # This also works for ICLR 2018
        # If it is from ICLR and it is not a year with decision field its a bit more tedious --
        # Instead in their 'Official Reviews' there is a field called 'rating'
        # The first character of the rating is a number 0-9 if it is larger than 5 it appears to be accepted.

        # Works for ICLR 2019
        if "recommendation" in note.content:
            return "Accept" in note.content["recommendation"]

        raise Exception("Cannot determine if a paper is accepted or not.")

    CONFERENCE_NAME_RE = re.compile("(?P<name>\\w+)\\.")

    @staticmethod
    def _conference_name(submission_note_url: str) -> str:
        return OpenReviewScraper.CONFERENCE_NAME_RE.search(submission_note_url).group("name")

    CONFERENCE_YEAR_RE = re.compile("(?P<year>\\d{4})")

    @staticmethod
    def _conference_year(submission_note_url: str) -> int:
        return OpenReviewScraper.CONFERENCE_YEAR_RE.search(submission_note_url).group("year")

    def __call__(self, sources: List[ScrapeURLs] = None,
                 num_threads: int = 1) -> Iterator[Paper]:
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
            decision_notes = {
                note.forum: self.__is_accepted(note)
                for note in openreview.tools.iterget_notes(
                    self.client, invitation=source.decision_notes)
            }
            number_decisions = len(decision_notes)
            self.logging.info(f"Downloaded {number_decisions} decision notes")
            self.logging.info(f"Downloading submission notes...")
            submission_notes = list(
                openreview.tools.iterget_notes(
                    self.client, invitation=source.submission_notes))
            number_submission = len(submission_notes)
            self.logging.info(
                f"Downloaded {number_submission} submission notes")

            num_accepted, num_rejected = 0, 0

            def downloader_fn(note: openreview.Note):
                # Ok with python threads
                nonlocal num_rejected, num_accepted, \
                    decision_notes, submission_notes

                paper = None
                try:
                    paper = Paper(
                        id=note.id,
                        authors=note.content["authors"],
                        title=note.content["title"],
                        abstract=note.content["abstract"],
                        pdf=self.client.get_pdf(note.id),
                        accepted=decision_notes[note.id],
                        conference=OpenReviewScraper._conference_name(source.submission_notes),
                        year=OpenReviewScraper._conference_year(source.submission_notes)
                    )

                    # This is okay when using the normal thread-pool since all threads
                    # run in the same process space & thanks to python's GIL we get no race-errors.
                    num_accepted += paper.accepted
                    num_rejected += (1 - paper.accepted)

                    return paper
                except openreview.openreview.OpenReviewException as e:
                    self.logging.warning(
                        f"Failed to create paper from {note.id}, reason: {str(e)}"
                    )
                except KeyError as e:
                    self.logging.warning(
                        f"Failed to create paper from {note.id}, missing field: {str(e)}"
                    )
                except MaxRetryError as e:
                    self.logging.warning(
                        f"Failed to create paper {note.content['title']} -- Reason: {e}"
                    )
                except ConnectionError as e:
                    self.logging.warning(
                        f"Failed to create paper {note.content['title']} -- Reason: {e}"
                    )
                except TimeoutError as e:
                    self.logging.warning(
                        f"Failed to create paper {note.content['title']} -- Reason: {e}"
                    )
                except Exception as e:
                    exception_type = type(e).__name__
                    self.logging.warning(
                        f"Unknown Exception {exception_type}: Failed to create paper {note.content['title']} -- Reason: {e}"
                    )

                return paper

            with ThreadPool(processes=num_threads) as pool:
                with tqdm(total=number_submission) as progress_bar:
                    i = 0
                    for note in pool.imap_unordered(downloader_fn,
                                                    submission_notes):
                        # update progress bar
                        progress_bar.update()

                        yield note

            time.sleep(10)

            self.logging.info(
                f"Accepted: {num_accepted} -- Rejected: {num_rejected}")


@dataclass
class ImagePaper:
    id: str
    authors: List[str]
    title: str
    abstract: str
    images: np.ndarray
    accepted: bool
    conference: str
    year: int


def convert_Paper_to_ImagePaper(paper: Paper, num_pages: int, image_width: int, image_height: int) -> Optional[
    ImagePaper]:
    """Convert the paper into a ImagePaper to be used by our vision model.

    This will extract the first page of the pdf and convert it into an image, all in memory.

    Args:
        paper: The paper dataclass containing the PDF information
        num_pages: Number of pages to include in paper images
        image_width: Width of the resulting images
        image_height: Height of the resulting images
    Returns:
        Optional[ImagePaper] - a dataclass containing a numpy array of width x height x images --  or None
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

        for front_page_index in range(pdf.numPages):
            front_page = pdf.getPage(front_page_index)

            if _is_valid_frontpage(front_page):
                break
        else:
            raise ValueError("No page is a valid front-page")

        # Now get raw pdf data
        pages_bytes = io.BytesIO()
        pdf_writer = PyPDF2.PdfFileWriter()

        # Add num_pages to pages bytes
        for page_index in range(num_pages):
            try:
                page = pdf.getPage(page_index + front_page_index)
                pdf_writer.addPage(page)
            except IndexError:
                break

        pdf_writer.write(pages_bytes)

        # pdf2image will create on image per page, here we resize and grayscale them
        np_images = \
            [
                np.array(
                    ImageOps.grayscale(
                        image.resize((image_width, image_height)))
                ) for image in pdf2image.convert_from_bytes(pages_bytes.getvalue())]

        # If np_images contains less than num_images we pad with black images
        padding = np.zeros((image_width, image_height))
        np_images = np_images + [padding] * (num_pages - len(np_images))
        np_images = [image.reshape((image_width, image_height, 1)) for image in np_images]
        np_images = np.concatenate(np_images, axis=2)

        return ImagePaper(id=paper.id,
                          authors=paper.authors,
                          title=paper.title,
                          abstract=paper.abstract,
                          images=np_images,
                          accepted=paper.accepted,
                          conference=paper.conference,
                          year=paper.year)
    except PdfReadError as e:
        logger.warning(
            f"PdfReadError: Unable to read {paper.title} reason: {e} -- discarding it"
        )
    except ValueError as e:
        logger.warning(
            f"ValueError: Paper {paper.title} was not converted to ImagePaper, Reason: {e}"
        )
    except TypeError as e:
        logger.warning(
            f"TypeError: Paper {paper.title} was not converted to ImagePaper, Reason: {e}"
        )
    except Exception as e:
        exception_type = type(e).__name__
        logger.warning(
            f"Unknown exception {exception_type}: {e} -- Dropping Paper {paper.title}"
        )


def ImagePapers_to_dataset(ipapers: Iterator[ImagePaper], base_path: str, image_infix: str = "images") -> None:
    """Takes a list of ImagePapers and constructs a csv with meta information and stores images as npy files.

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
        "conference": [],
        "year": [],
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
        meta_features["conference"].append(paper.conference)
        meta_features["year"].append(paper.year)

        relative_image_path = f"{image_infix}/{paper.id}"
        meta_features["image_path"].append(relative_image_path)

        absolute_image_path = base_path / relative_image_path
        # Write image to disk

        np.save(file=absolute_image_path, arr=paper.images)

    pd.DataFrame(meta_features).to_csv(base_path / "meta.csv", index=False)


def make_dataset(sources: List[ScrapeURLs],
                 dataset_base_path: str,
                 open_review_username: str,
                 open_review_password: str,
                 image_width: int,
                 image_height: int,
                 num_pages: int,
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
        image_width: Width of pages stored in dataset
        image_height: Height of pages stored in dataset
        num_pages: Number of pages stored per paper
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
    papers = (paper
              for paper in OpenReviewScraper(username=open_review_username,
                                             password=open_review_password)
              (sources=sources, num_threads=num_threads) if paper is not None)

    # Lazy converting papers
    ipapers = (paper for paper in map(
        lambda x: convert_Paper_to_ImagePaper(paper=x,
                                              num_pages=num_pages,
                                              image_width=image_width,
                                              image_height=image_height),
        papers) if paper is not None)

    ImagePapers_to_dataset(ipapers=ipapers,
                           base_path=dataset_base_path,
                           image_infix="images")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, help="Username to OpenReview", required=True)
    parser.add_argument("--password", type=str, help="Password to OpenReview", required=True)
    parser.add_argument("--base_path", type=str, help="Base path of dataset", required=True)
    parser.add_argument("--image_width", type=int, help="width of stored images", required=True)
    parser.add_argument("--image_height", type=int, help="height of stored images", required=True)
    parser.add_argument("--num_pages", type=int, help="number of pages stored per paper", required=True)
    parser.add_argument("--num_threads", type=int, help="Number of threads to run download in", default=5)

    args = parser.parse_args()

    logger.info(f"Making dataset at {args.base_path} with default sources")

    timestamp = time.time()
    make_dataset(sources=DEFAULT_SOURCES, dataset_base_path=args.base_path,
                 open_review_username=args.username, open_review_password=args.password,
                 image_width=args.image_width, image_height=args.image_height,
                 num_pages=args.num_pages, num_threads=args.num_threads)

    logger.info(
        f"Construction of dataset took {time.time() - timestamp} seconds")
