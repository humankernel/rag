from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from PIL import Image


class PdfPreprocessor:
    def __init__(self) -> None:
        config = {"output_format": "markdown", "output_dir": "docs/out"}
        config_parser = ConfigParser(config)
        self.pdf2md = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

    def convert(self, pdf_path: str) -> tuple[str, dict[str, Image.Image]]:
        doc_md = self.pdf2md(pdf_path)
        return (doc_md.markdown, doc_md.images)
