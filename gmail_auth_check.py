from gmail_utils import get_gmail_service, list_labels


def main():
    service = get_gmail_service()
    labels = list_labels(service)

    print("Gmail authentication successful.")
    print("Labels found:")
    for label in labels[:10]:
        print(f"- {label}")


if __name__ == "__main__":
    main()
