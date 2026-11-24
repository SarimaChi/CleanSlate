from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def convert(posts_xml: str, output_jsonl: str, site_url: str) -> int:
    """Convert StackExchange Posts.xml into accepted-answer instruction examples."""
    parsed_site_url = urlparse(site_url)
    if parsed_site_url.scheme not in {"http", "https"} or not parsed_site_url.netloc:
        raise ValueError("site_url must be an absolute http(s) URL for the StackExchange site")
    site_url = site_url.rstrip("/")
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    questions: dict[str, dict[str, str]] = {}
    accepted_ids: set[str] = set()
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for _, element in ET.iterparse(posts_xml, events=("end",)):
            if element.tag != "row":
                element.clear()
                continue
            post_type = element.attrib.get("PostTypeId")
            if post_type == "1":
                accepted = element.attrib.get("AcceptedAnswerId", "")
                if accepted:
                    accepted_ids.add(accepted)
                questions[element.attrib["Id"]] = {
                    "title": _plain_text(element.attrib.get("Title", "")),
                    "body": _plain_text(element.attrib.get("Body", "")),
                    "tags": element.attrib.get("Tags", ""),
                    "owner_user_id": element.attrib.get("OwnerUserId"),
                }
            elif post_type == "2":
                parent = questions.get(element.attrib.get("ParentId", ""))
                if parent and element.attrib.get("Id") in accepted_ids:
                    question = parent["title"]
                    if parent["body"]:
                        question += "\n\n" + parent["body"]
                    if parent["tags"]:
                        question += "\n\nTags: " + parent["tags"].replace("><", ", ").strip("<>")
                    row = {
                        "instruction": question,
                        "output": _plain_text(element.attrib.get("Body", "")),
                        "source_id": element.attrib["Id"],
                        "metadata": {
                            "source": "StackExchange",
                            "question_id": element.attrib.get("ParentId"),
                            "question_url": f"{site_url}/q/{element.attrib.get('ParentId')}",
                            "answer_url": f"{site_url}/a/{element.attrib['Id']}",
                            "question_owner_user_id": questions[element.attrib["ParentId"]].get("owner_user_id"),
                            "answer_owner_user_id": element.attrib.get("OwnerUserId"),
                        },
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            element.clear()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a StackExchange Posts.xml dump to JSONL.")
    parser.add_argument("--posts-xml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--site-url", required=True, help="Base URL of the source site, e.g. https://stackoverflow.com")
    args = parser.parse_args()
    print(f"Wrote {convert(args.posts_xml, args.output, args.site_url)} instruction-answer pairs")


if __name__ == "__main__":
    main()
