import openreview
import sys
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
from typing import Iterator, List, Optional, Tuple, TypeVar
import pathlib
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool, Pool
from enum import Enum
import matplotlib.pyplot as plt


class Mode(Enum):
    Download = "download"
    RGBFrontPage = "rgb-frontpage"
    GSFrontPage = "gs-frontpage"
    GSChannels = "gs-channels"
    RGBChannels = "rgb-channels"
    RGBBigImage = "rgb-bigimage"
    GSBigImage = "gs-bigimage"

    def __str__(self):
        return self.value


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

    # Seems to work, gives 598 rejected, 337 accepted
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
                    for note in pool.imap_unordered(downloader_fn,
                                                    submission_notes):
                        # update progress bar
                        progress_bar.update()

                        yield note

            time.sleep(10)

            self.logging.info(
                f"Accepted: {num_accepted} -- Rejected: {num_rejected}")


def download_pdfs(sources: List[ScrapeURLs],
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
    logger.info(f"Using {num_threads} threads")
    # Lazy downloading papers, Downloading all at once won't fit in memory.. just one ICML + MIDL takes more than
    # 30GB memory
    papers = (paper
              for paper in OpenReviewScraper(username=open_review_username,
                                             password=open_review_password)
              (sources=sources, num_threads=num_threads) if paper is not None)

    # Create dataset path if it does not exist
    base_path = pathlib.Path(dataset_base_path)
    if not base_path.is_dir():
        logger.warning(f"{base_path} is not a valid directory")
        logger.warning(f"Creating {base_path}")

        # Assume it works.. if someone can be bothered to add fail-safes, do so
        os.mkdir(base_path)

    papers_path = pathlib.Path(f"{dataset_base_path}/papers")

    # Create pdf folder if it does not exist
    if not papers_path.is_dir():
        logger.warning(f"{papers_path} is not a valid directory")
        logger.warning(f"Creating {papers_path}")

        # Assume it works.. if someone can be bothered to add fail-safes, do so
        os.mkdir(papers_path)

    meta_features = {
        "id": [],
        "authors": [],
        "title": [],
        "abstract": [],
        "accepted": [],
        "conference": [],
        "year": [],
        "paper_path": [],
    }

    logger.info(f"Storing papers at -- {base_path}")
    for paper in papers:
        # Add all meta features
        meta_features["id"].append(paper.id)
        meta_features["authors"].append("-".join(paper.authors))
        meta_features["title"].append(paper.title)
        meta_features["abstract"].append(paper.abstract)
        meta_features["accepted"].append(paper.accepted)
        meta_features["conference"].append(paper.conference)
        meta_features["year"].append(paper.year)

        relative_paper_path = f"papers/{paper.id}"
        meta_features["paper_path"].append(relative_paper_path)

        absolute_paper_path = base_path / relative_paper_path

        # Write pdf to disk
        with open(f"{absolute_paper_path}.pdf", "wb") as pdf_file:
            pdf_file.write(paper.pdf)

    pd.DataFrame(meta_features).to_csv(base_path / "meta.csv", index=False)


def pdf_loader(base_path: str):
    logger = logging.getLogger(LOGGER_NAME)

    base_path = pathlib.Path(base_path)
    if not base_path.is_dir():
        logger.fatal(f"{base_path} is not a directory")
        sys.exit(1)

    paper_path = base_path / "papers"

    if not paper_path.is_dir():
        logger.fatal(f"{paper_path} is not a directory")
        sys.exit(1)

    pdfs = list(paper_path.glob("*.pdf"))

    def _pdf_iterator():
        for paper in pdfs:
            yield paper

    return len(pdfs), _pdf_iterator()


def pdf_to_images(name: str, pdf: bytes, num_pages: int):
    """Convert the paper into a a list of images to be used by our vision model.

    This will extract the first page of the pdf and convert it into an image, all in memory.

    Args:
        name: Name of the pdf file
        pdf: Bytes of pdf file
        num_pages: Number of pages to include in paper images
    Returns:
        Optional[List[Image]]
    """
    logger = logging.getLogger(LOGGER_NAME)
    try:
        # A fake input file
        pdf_stream = io.BytesIO(pdf)

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

        # pdf2image will create on image per page
        return [image for image in pdf2image.convert_from_bytes(pages_bytes.getvalue())]

    except PdfReadError as e:
        logger.warning(
            f"PdfReadError: Unable to read paper {name} reason: {e} -- discarding it"
        )
    except ValueError as e:
        logger.warning(
            f"ValueError: Paper {name} was not converted to images, Reason: {e}"
        )
    except TypeError as e:
        logger.warning(
            f"TypeError:  Paper {name} was not converted to images, Reason: {e}"
        )
    except Exception as e:
        exception_type = type(e).__name__
        logger.warning(
            f"Unknown exception {exception_type}: {e} -- Dropping Paper {name}"
        )


def pdf_to_binary_blob(arguments: Tuple):
    dataset_base_path, num_pages, width, height, mode, skip_first_page, pdf_path = arguments
    """Loads pdfs and converts the to a binary blob

    Args:
        dataset_base_path: path to the base of a pdf dataset
        num_pages: Number of pages to include in paper images
        width: Width of the resulting images
        height: Height of the resulting images
        mode: The mode of image conversion, e.g GSChannels, RGBChannels..
        pdf_path: Path to a pdf file
    """
    # Inherit the variables from the outerscope

    # Load the PDF file
    with open(pdf_path, "rb") as pdf_file:
        name = pathlib.Path(pdf_path).stem
        pdf = pdf_file.read()

    if skip_first_page:
        # Skip the first page, but add a page to the end
        images = pdf_to_images(name=name, pdf=pdf, num_pages=num_pages + 1)
        images = images[1:]
    else:
        images = pdf_to_images(name=name, pdf=pdf, num_pages=num_pages)

    # If we were unable to extract images
    if images is None:
        return

    # For all images we will drop some pixels at the top because for some conferences
    # it says in the top if they were accepted or not

    images = np.array([np.array(image) for image in images])

    # Drops first 200 pixels from top
    images = images[:, 200:, :, :]
    images = [Image.fromarray(image) for image in images]

    # The binary blob is what will be written to the file
    # dataset_base_path/papers/name-mode-width-height.npy
    binary_blob = None

    def _get_grayscale():
        _binary_blob = [image.resize((width, height)) for image in images]
        _binary_blob = [ImageOps.grayscale(image) for image in _binary_blob]
        _binary_blob = [np.array(image) for image in _binary_blob]

        padding = np.zeros((width, height))

        # Pad blob so that all have num_pages images
        _binary_blob = _binary_blob + ([padding] * (num_pages - len(_binary_blob)))
        _binary_blob = np.array(_binary_blob)
        return _binary_blob

    def _get_rgb():
        _binary_blob = [image.resize((width, height)) for image in images]
        _binary_blob = [np.array(image) for image in _binary_blob]

        padding = np.zeros((width, height, 3))

        # Pad blob so that all have num_pages images
        _binary_blob = _binary_blob + ([padding] * (num_pages - len(_binary_blob)))
        _binary_blob = np.array(_binary_blob)
        return _binary_blob

    if mode == Mode.GSChannels or mode == Mode.GSFrontPage:
        binary_blob = _get_grayscale()
    if mode == Mode.RGBChannels or mode == Mode.RGBFrontPage:
        binary_blob = _get_rgb()
    if mode == Mode.RGBBigImage:
        assert (num_pages == 8)  # NOTE: This mode only works with 8 pages. Can easily be extended
        binary_blob = _get_rgb()

        binary_blob_top = np.hstack(binary_blob[0:4])
        binary_blob_bottom = np.hstack(binary_blob[4:])
        binary_blob = np.vstack([binary_blob_top, binary_blob_bottom])
    if mode == Mode.GSBigImage:
        assert (num_pages == 8)  # NOTE: This mode only works with 8 pages. Can easily be extended
        binary_blob = _get_grayscale()

        binary_blob_top = np.hstack(binary_blob[0:4])
        binary_blob_bottom = np.hstack(binary_blob[4:])
        binary_blob = np.vstack([binary_blob_top, binary_blob_bottom])

    binary_blob_path = f"{dataset_base_path}/papers/{name}-{mode}-{width}-{height}"
    np.save(binary_blob_path, binary_blob)


def convert_pdf_dataset(dataset_base_path: str,
                        num_pages: int,
                        width: int,
                        height: int,
                        mode: Mode,
                        num_processes: int,
                        skip_first_page: bool):
    """Convert the pdf dataset into a different representation, e.g grayscaled images
    Args:
        dataset_base_path: path to the base of a pdf dataset
        num_pages: Number of pages to include in paper images
        width: Width of the resulting images
        height: Height of the resulting images
        mode: The mode of image conversion, e.g GSChannels, RGBChannels..
        num_processes: The number of processes to use in dataset creation
    Returns:
        Nothing, this writes to the dataset_base_path/papers directory
    """

    with Pool(processes=num_processes) as pool:
        num_pdfs, pdf_iterator = pdf_loader(base_path=dataset_base_path)

        jobs = ((dataset_base_path, num_pages, width, height, mode, skip_first_page, pdf_path)
                for pdf_path in pdf_iterator)

        with tqdm(total=num_pdfs) as progress_bar:
            for _ in pool.imap_unordered(pdf_to_binary_blob, jobs, chunksize=5):
                progress_bar.update()



def ensure_non_null(name: str, arg):
    if arg is None:
        logger.fatal(f"Missing argument {name}, exiting...")
        sys.exit(1)

    return arg

def inspect_binary_blob(path_to_blob: str, mode: Mode):
    """Visualize a binary blob

    Args:
        path_to_blob: file path to the binary blob

    Returns:
        Nothing, will plot
    """
    logger = logging.getLogger(LOGGER_NAME)
    path_to_blob = pathlib.Path(path_to_blob)

    if not path_to_blob.is_file():
        logger.fatal(f"{path_to_blob} is not a valid file")
        sys.exit(1)

    binary_blob = np.load(path_to_blob)
    if mode == Mode.RGBBigImage:
        plt.figure(figsize=(10, 6))
        print(binary_blob.shape)
        plt.imshow(binary_blob)
        plt.savefig(f"{path_to_blob}.png")
    elif mode == Mode.RGBChannels:
        plt.figure(figsize=(10, 6))

        index = 1
        for i in range(4):
            for j in range(2):
                plt.subplot(2, 4, index)
                plt.imshow(binary_blob[index - 1])
                plt.axis("off")

                index += 1

        plt.savefig(f"{path_to_blob}.png")
    elif mode == Mode.RGBFrontPage:
        plt.figure(figsize=(10, 6))
        plt.imshow(binary_blob[0])
        plt.axis("off")
        plt.savefig(f"{path_to_blob}.png")
    else:
        raise NotImplementedError(f"{mode} is not implemented yet")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, help="Username to OpenReview")
    parser.add_argument("--password", type=str, help="Password to OpenReview")
    parser.add_argument("--base_path", type=str, help="Base path of dataset")
    parser.add_argument("--image_width", type=int, help="width of stored images")
    parser.add_argument("--image_height", type=int, help="height of stored images")
    parser.add_argument("--num_pages", type=int, help="number of pages stored per paper")
    parser.add_argument("--num_threads", type=int, help="Number of threads to run download in")
    parser.add_argument("--num_processes", type=int, help="Number of processes to run dataset conversion in")
    parser.add_argument("--mode", type=Mode, choices=list(Mode))
    parser.add_argument('--skip_first_page', action='store_true')  # Skip the first page, but add a page to the end
    parser.add_argument("--inspect", type=str, help="A file to inspect")

    args = parser.parse_args()




    timestamp = time.time()

    # If we run inspection we'll exit afterwards
    if args.inspect:
        logger.info(f"Inspecting {args.inspect}")
        inspect_binary_blob(path_to_blob=args.inspect, mode=args.mode)
        sys.exit(0)

    if args.mode == Mode.Download:
        logger.info(f"Making dataset at {args.base_path} with default sources")
        download_pdfs(sources=DEFAULT_SOURCES,
                      dataset_base_path=ensure_non_null("base_path", args.base_path),
                      open_review_username=ensure_non_null("username", args.username),
                      open_review_password=ensure_non_null("password", args.password),
                      num_threads=ensure_non_null("num_threads", args.num_threads))

    if args.mode == Mode.GSChannels:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", args.num_pages),
                            width=ensure_non_null("image_width", args.image_width),
                            height=ensure_non_null("image_height", args.image_height),
                            mode=Mode.GSChannels,
                            num_processes=ensure_non_null("num_processes", args.num_processes),
                            skip_first_page=ensure_non_null("skip_first_page", args.skip_first_page))

    if args.mode == Mode.RGBChannels:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", args.num_pages),
                            width=ensure_non_null("image_width", args.image_width),
                            height=ensure_non_null("image_height", args.image_height),
                            mode=Mode.RGBChannels,
                            num_processes=ensure_non_null("num_processes", args.num_processes),
                            skip_first_page=ensure_non_null("skip_first_page", args.skip_first_page))

    if args.mode == Mode.GSBigImage:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", args.num_pages),
                            width=ensure_non_null("image_width", args.image_width),
                            height=ensure_non_null("image_height", args.image_height),
                            mode=Mode.GSBigImage,
                            num_processes=ensure_non_null("num_processes", args.num_processes),
                            skip_first_page=ensure_non_null("skip_first_page", args.skip_first_page))

    if args.mode == Mode.RGBBigImage:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            num_pages=ensure_non_null("num_pages", args.num_pages),
                            width=ensure_non_null("image_width", args.image_width),
                            height=ensure_non_null("image_height", args.image_height),
                            mode=Mode.RGBBigImage,
                            num_processes=ensure_non_null("num_processes", args.num_processes),
                            skip_first_page=ensure_non_null("skip_first_page", args.skip_first_page))

    if args.mode == Mode.GSFrontPage:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            # Gets only frontpage
                            num_pages=1,
                            width=ensure_non_null("image_width", args.image_width),
                            height=ensure_non_null("image_height", args.image_height),
                            mode=Mode.GSFrontPage,
                            num_processes=ensure_non_null("num_processes", args.num_processes),
                            skip_first_page=ensure_non_null("skip_first_page", args.skip_first_page))

    if args.mode == Mode.RGBFrontPage:
        convert_pdf_dataset(dataset_base_path=ensure_non_null("base_path", args.base_path),
                            # Gets only frontpage
                            num_pages=1,
                            width=ensure_non_null("image_width", args.image_width),
                            height=ensure_non_null("image_height", args.image_height),
                            mode=Mode.RGBFrontPage,
                            num_processes=ensure_non_null("num_processes", args.num_processes),
                            skip_first_page=ensure_non_null("skip_first_page", args.skip_first_page))

    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")
