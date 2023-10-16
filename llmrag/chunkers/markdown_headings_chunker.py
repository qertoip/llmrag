import re

from tools import kb_path


class MarkdownHeadingsChunker:
    """
    Chunks Markdown documents by major headings (# Heading 1, ## Heading 2).

    Such naive and coarse-grained chunking may be problematic,
    but we need to start somewhere to get things done.
    """

    empty_anchor_regex = re.compile(r'<a[^>]*>\s*</a>')

    def chunk(self, markdown_doc: str) -> list[str]:
        clean_doc = self.cleanup(markdown_doc)
        chunks = re.split(r'\n#{1,2}\s', clean_doc)
        chunks = [c.strip() for c in chunks]
        chunks = [c for c in chunks if len(c.splitlines()) >= 2]  # only headings with content
        chunks = [c for c in chunks if len(c.split()) >= 8]       # only headings+content with at least 10 words
        chunks = [c for c in chunks if len(c) >= 50]              # only headings+content with at least 50 chars
        chunks = [self.standardize_header(c) for c in chunks]
        return chunks

    def cleanup(self, doc: str):
        doc = doc.strip()
        doc = self.remove_empty_anchors(doc)
        doc = self.remove_useless_backslash(doc)
        return doc

    def remove_empty_anchors(self, markdown_doc: str):
        return re.sub(self.empty_anchor_regex, '', markdown_doc)

    def remove_useless_backslash(self, doc: str) -> str:
        return doc.\
                   replace('\.', '.').\
                   replace('\(', '(').\
                   replace('\)', ')').\
                   replace('\_', '_').\
                   replace('\-', '-')

    def standardize_header(self, chunk: str):
        if chunk.startswith('# '):
            chunk = chunk[2:]  # Remove Heading 1
        chunk = '## ' + chunk  # Ensure Heading 2
        return chunk


if __name__ == '__main__':
    doc = (kb_path() / 'aws-properties-sagemaker-app-resourcespec.md').read_text()
    chunks = MarkdownHeadingsChunker().chunk(doc)
    recombined = '\n-------------------------------------------------------------------------------------------\n'.join(chunks)
    print(recombined)
