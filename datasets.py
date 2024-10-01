import os
import requests
from urllib.parse import urlparse
import zipfile
import tarfile


class DatasetDownloader:
    def __init__(self, datasets, download_dir=None):
        """
        Initializes the DatasetDownloader.

        :param datasets: Dictionary or list of datasets to download.
                         - For direct links: Provide a dictionary with names as keys and URLs as values.
                         - For Zenodo datasets: Provide a list of URLs.
        :param download_dir: (Optional) Directory where datasets will be downloaded and extracted.
                             Defaults to 'datasets/Segmentation' if not provided.
        """
        self.datasets = datasets
        # Set default download directory if none provided
        if download_dir is None:
            self.download_dir = os.path.join('datasets', 'Segmentation')
        else:
            self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_and_extract(self):
        if isinstance(self.datasets, dict):
            # Handle datasets provided as a dictionary (name: url)
            for name, url in self.datasets.items():
                self._download_and_extract_direct(name, url)
        elif isinstance(self.datasets, list):
            # Handle datasets provided as a list of Zenodo URLs
            for url in self.datasets:
                self._download_and_extract_zenodo(url)
        else:
            print("Datasets should be either a list of URLs or a dictionary of name: url pairs.")

    def _download_and_extract_direct(self, name, url):
        # Download the file
        print(f"Downloading {name} from {url}")
        r = requests.get(url, stream=True, verify=False)
        if r.status_code != 200:
            print(f"Failed to download {name}")
            return
        file_name = os.path.basename(urlparse(url).path)
        file_path = os.path.join(self.download_dir, file_name)
        # Write content to file
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Extract the file if it's an archive
        self._extract_file(file_path)

    def _download_and_extract_zenodo(self, url):
        record_id = self._get_record_id(url)
        if not record_id:
            print(f"Could not parse record ID from URL: {url}")
            return

        files = self._get_files_from_record(record_id)
        if not files:
            print(f"No files found for record {record_id}")
            return

        for file_info in files:
            file_name = file_info['key']
            download_link = file_info['links']['self']

            # Download the file
            print(f"Downloading {file_name} from {download_link}")
            r = requests.get(download_link, stream=True, verify=False)
            if r.status_code != 200:
                print(f"Failed to download file {file_name}")
                continue
            file_path = os.path.join(self.download_dir, file_name)
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # Extract the file if it's an archive
            self._extract_file(file_path)

    def _get_record_id(self, url):
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        if 'record' in path_parts:
            idx = path_parts.index('record')
        elif 'records' in path_parts:
            idx = path_parts.index('records')
        else:
            return None
        return path_parts[idx + 1]

    def _get_files_from_record(self, record_id):
        api_url = f'https://zenodo.org/api/records/{record_id}'
        response = requests.get(api_url, verify=False)
        if response.status_code != 200:
            print(f"Failed to get metadata for record {record_id}")
            return None
        data = response.json()
        return data.get('files', [])

    def _extract_file(self, file_path):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        # Handle double extensions like .tar.gz and .tar.bz2
        if file_name.endswith(('.tar.gz', '.tgz', '.tar.bz2', '.tar.xz')):
            base_name = os.path.splitext(base_name)[0]
        extract_dir = os.path.join(self.download_dir, base_name)
        os.makedirs(extract_dir, exist_ok=True)
        # Extract the file if it's an archive
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check if the zip contains a single top-level folder
                top_level_dirs = set()
                for member in zip_ref.namelist():
                    # Ignore directory entries
                    if member.endswith('/'):
                        continue
                    parts = member.split('/')
                    if len(parts) > 1:
                        top_level_dirs.add(parts[0])
                    else:
                        top_level_dirs.add('.')
                if len(top_level_dirs) == 1 and '.' not in top_level_dirs:
                    # Extract as is
                    zip_ref.extractall(self.download_dir)
                else:
                    # Extract into the new subfolder
                    zip_ref.extractall(extract_dir)
            print(f"Extracted {file_name} into {extract_dir}")
            os.remove(file_path)  # Remove the zip file after extraction
        elif file_name.endswith(('.tar.gz', '.tgz', '.tar', '.tar.bz2', '.tar.xz')):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                # Check if the tar contains a single top-level folder
                top_level_dirs = set()
                for member in tar_ref.getmembers():
                    if member.isdir():
                        continue
                    parts = member.name.split('/')
                    if len(parts) > 1:
                        top_level_dirs.add(parts[0])
                    else:
                        top_level_dirs.add('.')
                if len(top_level_dirs) == 1 and '.' not in top_level_dirs:
                    # Extract as is
                    tar_ref.extractall(self.download_dir)
                else:
                    # Extract into the new subfolder
                    tar_ref.extractall(extract_dir)
            print(f"Extracted {file_name} into {extract_dir}")
            os.remove(file_path)  # Remove the tar file after extraction
        else:
            print(f"File {file_name} is not an archive or unsupported format, skipping extraction.")


# Usage example:
if __name__ == '__main__':

    # Dictionary of datasets to be placed in 'datasets/Classification'
    datasets = {
        'Acinetobacter baumanii': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Acinetobacter.baumanii.zip',
        'Actinomyces israelii': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Actinomyces.israeli.zip',
        'Bacteroides fragilis': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Bacteroides.fragilis.zip',
        'Bifidobacterium spp.': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Bifidobacterium.spp.zip',
        'Candida albicans': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Candida.albicans.zip',
        'Clostridium perfringens': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Clostridium.perfringens.zip',
        'Enterococcus faecium': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Enterococcus.faecium.zip',
        'Enterococcus faecalis': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Enterococcus.faecalis.zip',
        'Escherichia coli': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Escherichia.coli.zip',
        'Fusobacterium spp.': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Fusobacterium.zip',
        'Lactobacillus casei': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.casei.zip',
        'Lactobacillus crispatus': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.crispatus.zip',
        'Lactobacillus delbrueckii': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.delbrueckii.zip',
        'Lactobacillus gasseri': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.gasseri.zip',
        'Lactobacillus jehnsenii': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.jehnsenii.zip',
        'Lactobacillus johnsonii': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.johnsonii.zip',
        'Lactobacillus paracasei': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.paracasei.zip',
        'Lactobacillus plantaru': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.plantarum.zip',
        'Lactobacillus reuteri': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.reuteri.zip',
        'Lactobacillus rhamnosus': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.rhamnosus.zip',
        'Lactobacillus salivarius': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Lactobacillus.salivarius.zip',
        'Listeria monocytogenes': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Listeria.monocytogenes.zip',
        'Micrococcus spp.': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Micrococcus.spp.zip',
        'Neisseria gonorrhoeae': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Neisseria.gonorrhoeae.zip',
        'Porphyromonas gingivalis': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Porfyromonas.gingivalis.zip',
        'Propionibacterium acnes': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Propionibacterium.acnes.zip',
        'Proteus spp.': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Proteus.zip',
        'Pseudomonas aeruginosa': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Pseudomonas.aeruginosa.zip',
        'Staphylococcus aureus': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.aureus.zip',
        'Staphylococcus epidermidis': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.epidermidis.zip',
        'Staphylococcus saprophiticus': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Staphylococcus.saprophiticus.zip',
        'Streptococcus agalactiae': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Streptococcus.agalactiae.zip',
        'Veionella spp.': 'https://doctoral.matinf.uj.edu.pl/database/dibas/Veionella.zip'
    }

    download_dir = os.path.join('datasets', 'Classification')
    downloader = DatasetDownloader(datasets, download_dir=download_dir)
    downloader.download_and_extract()

    urls_segmentation = [
        'https://zenodo.org/records/5550933#.YYMBMxwxlqg',
        'https://zenodo.org/records/5550935#.YYMBAxwxlqg',
        'https://zenodo.org/records/5550968#.YYMBgRwxlqg',
        'https://zenodo.org/records/5639253#.YYMAeBwxlqg',
        'https://zenodo.org/records/5551009#.YYMBthwxlqg',
    ]

    # specify a directory
    directory_segmentation = 'datasets/Segmentation'

    # Create an instance with a custom download directory
    downloader = DatasetDownloader(urls_segmentation, download_dir=directory_segmentation)
    downloader.download_and_extract()

    urls_object_detection = [
        'https://zenodo.org/records/5551016#.YYMCjhwxlqg',
        'https://zenodo.org/records/5551057#.YYMCtRwxlqg',
    ]

    # specify a directory
    directory_object_detection = 'datasets/ObjectDetection'

    # Create an instance with a custom download directory
    downloader = DatasetDownloader(urls_object_detection, download_dir=directory_object_detection)
    downloader.download_and_extract()




