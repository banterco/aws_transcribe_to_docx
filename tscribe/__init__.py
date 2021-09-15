""" Transform AWS Transcribe json files to docx, csv, sqlite and vtt. """

import json, datetime
import statistics
from pathlib import Path
from time import perf_counter
import pandas
import sqlite3
import webvtt
import logging


def convert_time_stamp(timestamp: str) -> str:
    """ Function to help convert timestamps from s to H:M:S """
    delta = datetime.timedelta(seconds=float(timestamp))
    seconds = delta - datetime.timedelta(microseconds=delta.microseconds)
    return str(seconds)


def load_json_as_dict(filepath: str) -> dict:
    """Load in JSON file and return as dict"""
    logging.info("Loading json")

    json_filepath = Path(filepath)
    assert json_filepath.is_file(), "JSON file does not exist"

    data = json.load(open(json_filepath.absolute(), "r", encoding="utf-8"))
    assert "jobName" in data
    assert "results" in data
    assert "status" in data

    assert data["status"] == "COMPLETED", "JSON file not shown as completed."

    logging.debug("json checks psased")
    return data


def calculate_confidence_statistics(data: dict) -> dict:
    """Confidence Statistics"""
    logging.info("Gathering confidence statistics")

    # Stats dictionary
    stats = {
        "timestamps": [],
        "accuracy": [],
        "9.8": 0,
        "9": 0,
        "8": 0,
        "7": 0,
        "6": 0,
        "5": 0,
        "4": 0,
        "3": 0,
        "2": 0,
        "1": 0,
        "0": 0,
        "total": len(data["results"]["items"]),
    }

    # Confidence count
    for item in data["results"]["items"]:
        if item["type"] == "pronunciation":

            stats["timestamps"].append(float(item["start_time"]))

            confidence_decimal = float(item["alternatives"][0]["confidence"])
            confidence_integer = int(confidence_decimal * 100)

            stats["accuracy"].append(confidence_integer)

            if confidence_decimal >= 0.98:
                stats["9.8"] += 1
            else:
                rough_confidence = str(int(confidence_decimal * 10))
                stats[rough_confidence] += 1

    return stats


def decode_transcript_to_dataframe(data: str):
    """Decode the transcript into a pandas dataframe"""
    logging.info("Decoding transcript")

    decoded_data = {"start_time": [], "end_time": [], "speaker": [], "comment": []}

    # If speaker identification
    if "speaker_labels" in data["results"].keys():
        logging.debug("Transcipt has speaker_labels")

        # A segment is a blob of pronounciation and punctuation by an individual speaker
        for segment in data["results"]["speaker_labels"]["segments"]:

            # If there is content in the segment, add a row, write the time and speaker
            if len(segment["items"]) > 0:
                decoded_data["start_time"].append(
                    convert_time_stamp(segment["start_time"])
                )
                decoded_data["end_time"].append(convert_time_stamp(segment["end_time"]))
                decoded_data["speaker"].append(segment["speaker_label"])
                decoded_data["comment"].append("")

                # For each word in the segment...
                for word in segment["items"]:

                    # Get the word with the highest confidence
                    pronunciations = list(
                        filter(
                            lambda x: x["type"] == "pronunciation",
                            data["results"]["items"],
                        )
                    )
                    word_result = list(
                        filter(
                            lambda x: x["start_time"] == word["start_time"]
                            and x["end_time"] == word["end_time"],
                            pronunciations,
                        )
                    )
                    result = sorted(
                        word_result[-1]["alternatives"], key=lambda x: x["confidence"]
                    )[-1]

                    # Write the word
                    decoded_data["comment"][-1] += " " + result["content"]

                    # If the next item is punctuation, write it
                    try:
                        word_result_index = data["results"]["items"].index(
                            word_result[0]
                        )
                        next_item = data["results"]["items"][word_result_index + 1]
                        if next_item["type"] == "punctuation":
                            decoded_data["comment"][-1] += next_item["alternatives"][0][
                                "content"
                            ]
                    except IndexError:
                        pass

    # If channel identification
    elif "channel_labels" in data["results"].keys():
        logging.debug("Transcipt has channel_labels")

        # For each word in the results
        for word in data["results"]["items"]:

            # Punctuation items do not include a start_time
            if "start_time" not in word.keys():
                continue

            # Identify the channel
            channel = list(
                filter(
                    lambda x: word in x["items"],
                    data["results"]["channel_labels"]["channels"],
                )
            )[0]["channel_label"]

            # If still on the same channel, add the current word to the line
            if (
                channel in decoded_data["speaker"]
                and decoded_data["speaker"][-1] == channel
            ):
                current_word = sorted(
                    word["alternatives"], key=lambda x: x["confidence"]
                )[-1]
                decoded_data["comment"][-1] += " " + current_word["content"]

            # Else start a new line
            else:
                decoded_data["start_time"].append(
                    convert_time_stamp(word["start_time"])
                )
                decoded_data["end_time"].append(convert_time_stamp(word["end_time"]))
                decoded_data["speaker"].append(channel)
                current_word = sorted(
                    word["alternatives"], key=lambda x: x["confidence"]
                )[-1]
                decoded_data["comment"].append(current_word["content"])

            # If the next item is punctuation, write it
            try:
                word_result_index = data["results"]["items"].index(word)
                next_item = data["results"]["items"][word_result_index + 1]
                if next_item["type"] == "punctuation":
                    decoded_data["comment"][-1] += next_item["alternatives"][0][
                        "content"
                    ]
            except IndexError:
                pass

    # Neither speaker nor channel identification
    else:
        logging.debug("No speaker_labels or channel_labels")

        decoded_data["start_time"] = convert_time_stamp(
            list(
                filter(lambda x: x["type"] == "pronunciation", data["results"]["items"])
            )[0]["start_time"]
        )
        decoded_data["end_time"] = convert_time_stamp(
            list(
                filter(lambda x: x["type"] == "pronunciation", data["results"]["items"])
            )[-1]["end_time"]
        )
        decoded_data["speaker"].append("")
        decoded_data["comment"].append("")

        # Add words
        for word in data["results"]["items"]:

            # Get the word with the highest confidence
            result = sorted(word["alternatives"], key=lambda x: x["confidence"])[-1]

            # Write the word
            decoded_data["comment"][-1] += " " + result["content"]

            # If the next item is punctuation, write it
            try:
                word_result_index = data["results"]["items"].index(word)
                next_item = data["results"]["items"][word_result_index + 1]
                if next_item["type"] == "punctuation":
                    decoded_data["comment"][-1] += next_item["alternatives"][0][
                        "content"
                    ]
            except IndexError:
                pass

    # Produce pandas dataframe
    dataframe = pandas.DataFrame(
        decoded_data, columns=["start_time", "end_time", "speaker", "comment"]
    )

    # Clean leading whitespace
    dataframe["comment"] = dataframe["comment"].str.lstrip()

    return dataframe


def write_vtt(dataframe, filename):
    """Output to VTT format"""
    logging.info("Writing VTT")

    # Initialize vtt
    vtt = webvtt.WebVTT()

    # Iterate through dataframe
    for _, row in dataframe.iterrows():

        # If the segment has 80 or less characters
        if len(row["comment"]) <= 80:

            caption = webvtt.Caption(
                start=row["start_time"] + ".000",
                end=row["end_time"] + ".000",
                text=row["comment"],
            )

        # If the segment has more than 80 characters, use lines
        else:

            lines = []
            text = row["comment"]

            while len(text) > 80:
                text = text.lstrip()
                last_space = text[:80].rindex(" ")
                lines.append(text[:last_space])
                text = text[last_space:]

            caption = webvtt.Caption(
                row["start_time"] + ".000", row["end_time"] + ".000", lines
            )

        if row["speaker"]:
            caption.identifier = row["speaker"]

        vtt.captions.append(caption)

    vtt.save(filename)
    logging.info("VTT saved to %s", filename)


def write(transcript_filepath, **kwargs):
    """Main function, write transcript file from json"""

    # Performance timer start
    start = perf_counter()
    logging.info("=" * 32)
    logging.debug("Started at %s", start)
    logging.info("Source file: %s", transcript_filepath)
    logging.debug("kwargs = %s", str(kwargs))

    # Load json file as dict
    data = load_json_as_dict(transcript_filepath)

    # Decode transcript
    dataframe = decode_transcript_to_dataframe(data)

    # Output
    output_format = kwargs.get("format", "docx")

    # Deprecated tmp_dir by improving save_as
    if kwargs.get("tmp_dir"):
        logging.warning("tmp_dir in kwargs")
        raise Exception("tmp_dir has been deprecated, use save_as instead")

    # Output to docx (default behaviour)
    if output_format == "docx":
        output_filepath = kwargs.get(
            "save_as", Path(transcript_filepath).with_suffix(".docx")
        )
        write_docx(data, output_filepath)

    # Output to CSV
    elif output_format == "csv":
        output_filepath = kwargs.get(
            "save_as", Path(transcript_filepath).with_suffix(".csv")
        )
        dataframe.to_csv(output_filepath)

    # Output to sqlite
    elif output_format == "sqlite":
        output_filepath = kwargs.get(
            "save_as", Path(transcript_filepath).with_suffix(".db")
        )
        conn = sqlite3.connect(str(output_filepath))
        dataframe.to_sql("transcript", conn)
        conn.close()

    # Output to VTT
    elif output_format == "vtt":
        output_filepath = kwargs.get(
            "save_as", Path(transcript_filepath).with_suffix(".vtt")
        )
        write_vtt(dataframe, output_filepath)

    else:
        raise Exception("Output format should be 'docx', 'csv', 'sqlite' or 'vtt'")

    # Performance timer finish
    finish = perf_counter()
    logging.debug("Finished at %s", finish)
    duration = round(finish - start, 2)

    print(f"{output_filepath} written in {duration} seconds.")
    logging.info("%s written in %s seconds.", output_filepath, duration)
