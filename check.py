import urllib.error
import urllib.request
from http.client import HTTPResponse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

URL = 'http://vision.dpieczynski.pl:8080'


def main():
    student_id = 131279 # Wypełnić
    python_file_path = Path('main.py')  # Wypełnić
    vocabulary_file_path = Path('vocab_model.p')  # Wypełnić
    classifier_file_path = Path('clf.p')  # Wypełnić

    data = BytesIO()
    with ZipFile(data, mode='w') as zip_file:
        zip_file.write(python_file_path, 'main.py')
        zip_file.write(vocabulary_file_path, 'vocab_model.p')
        zip_file.write(classifier_file_path, 'clf.p')

    data.seek(0)

    try:
        response: HTTPResponse = urllib.request.urlopen(f'{URL}/{student_id}', data.read())
        print(response.read().decode())
    except urllib.error.HTTPError as e:
        print(e.read().decode())


if __name__ == '__main__':
    main()