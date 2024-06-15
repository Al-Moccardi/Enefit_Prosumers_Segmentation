import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import textwrap
import os
import zipfile
import psutil
import pandas as pd
import json
import matplotlib.pyplot as plt
import chardet
import csv

class Inspector:
    def __init__(self, folder_path, report_title="Data Inspection Report", report_subtitle="Overview of Dataset Quality and Statistics", custom_text=None):
        self.folder_path = folder_path
        self.report_title = report_title
        self.report_subtitle = report_subtitle
        self.custom_text = custom_text or f"Generated on: {datetime.now().strftime('%Y-%m-%d')}"
        self.report_path = os.path.join(self.folder_path, 'data_inspection_report.pdf')
        
    def add_custom_text(self, ax, text, fontsize=10, position=(0.5, 0.5)):
        ax.text(position[0], position[1], text, fontsize=fontsize, transform=ax.transAxes, ha='center', va='center')
        
    def format_table_for_plot(self, ax, table):
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            if key[0] == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')

    def evaluate_data_quality(self, metrics):
        score = 100
        score -= 20 if metrics['common_time_range'] == 'No' else 0
        score -= 10 * len(metrics['granularity_issues'])  # Deduct 10 points for each mismatch in granularity
        score -= 20 if metrics['total_null_values'] / metrics['total_rows'] > 0.1 else 0
        score -= 20 if metrics['total_duplicate_rows'] / metrics['total_rows'] > 0.05 else 0
        score -= 5 * len(metrics['duplicated_columns'])  # Deduct 5 points for each duplicated column

        if score >= 80:
            return 'Very Good', score
        elif score >= 60:
            return 'Good', score
        elif score >= 40:
            return 'Normal', score
        elif score >= 20:
            return 'Bad', score
        else:
            return 'Very Bad', score

    def wrap_text(self, text, width=30):
        return '\n'.join(textwrap.wrap(text, width))

    def generate_report(self):
        total_null_values = 0
        total_duplicate_rows = 0
        total_rows = 0
        datetime_columns = {}
        duplicated_columns = set()
        issue_datasets = {}
        granularity_issue_files = set()
        datatype_summary = {}
        time_ranges = {}
        

        with PdfPages(self.report_path) as pdf:
            plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            ax = plt.gca()
            plt.axis('off')
            self.add_custom_text(ax, self.report_title, fontsize=18, position=(0.5, 0.75))
            self.add_custom_text(ax, self.report_subtitle, fontsize=14, position=(0.5, 0.70))
            self.add_custom_text(ax, self.custom_text, fontsize=10, position=(0.5, 0.60))
            pdf.savefig()
            plt.close()
            
            null_values_stats = []
            datasets_processed = 0
            

            

            for filename in os.listdir(self.folder_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(self.folder_path, filename)
                    data = pd.read_csv(file_path)

                    # Automatically convert columns that might contain datetime information
                    for col in data.columns:
                        if 'time' in col.lower() or 'date' in col.lower():
                            data[col] = pd.to_datetime(data[col], errors='coerce')

                    date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                    time_ranges[filename] = {col: (data[col].min(), data[col].max()) for col in date_columns if pd.to_datetime(data[col], errors='coerce').notna().any()}

                    datatype_counts = data.dtypes.apply(lambda x: x.name).value_counts()
                    for dtype, count in datatype_counts.items():
                        if dtype in datatype_summary:
                            datatype_summary[dtype] += count
                        else:
                            datatype_summary[dtype] = count

                    total_rows += data.shape[0]
                    null_count = data.isnull().sum().sum()
                    total_null_values += null_count
                    duplicate_count = data.duplicated().sum()
                    total_duplicate_rows += duplicate_count
                    duplicated_columns.update(data.columns[data.columns.duplicated()])

                    null_values_stats.append(null_count)

                    if null_count > 0:
                        issue_datasets.setdefault('Null Values', []).append(filename)
                    if duplicate_count > 0:
                        issue_datasets.setdefault('Duplicate Rows', []).append(filename)

                    if datasets_processed % 2 == 0:
                        plt.figure(figsize=(11.69, 8.27))  # A4 landscape
                        ax = plt.gca()
                        plt.axis('off')

                    ax_table = plt.subplot(2, 1, (datasets_processed % 2) + 1)
                    ax_table.axis('off')
                    plt.title(f'{filename} - Detailed Report', fontsize=16)

                    file_info = pd.DataFrame({
                        'Metric': ['Total Rows', 'Total Columns', 'Complete Cases', 'Duplicated Rows', 'Missing Values'],
                        'Value': [data.shape[0], data.shape[1], data.notnull().all(axis=1).sum(), duplicate_count, null_count]
                    })

                    table = ax_table.table(cellText=file_info.values, colLabels=file_info.columns, loc='center')
                    self.format_table_for_plot(ax_table, table)

                    if datasets_processed % 2 == 1 or filename == os.listdir(self.folder_path)[-1]:
                        pdf.savefig()
                        plt.close()

                    datasets_processed += 1

            for filename in os.listdir(self.folder_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(self.folder_path, filename)
                    data = pd.read_csv(file_path)

                    # Automatically convert columns that might contain datetime information
                    for col in data.columns:
                        if 'time' in col.lower() or 'date' in col.lower():
                            data[col] = pd.to_datetime(data[col], errors='coerce')

                    total_rows += data.shape[0]
                    null_count = data.isnull().sum().sum()
                    total_null_values += null_count
                    duplicate_count = data.duplicated().sum()
                    total_duplicate_rows += duplicate_count
                    duplicated_columns.update(data.columns[data.columns.duplicated()])

                    # Log issues and affected datasets
                    if null_count > 0:
                        issue_datasets.setdefault('Null Values', []).append(filename)
                    if duplicate_count > 0:
                        issue_datasets.setdefault('Duplicate Rows', []).append(filename)

                    # Time range and granularity analysis
                    for col in data.select_dtypes(include=['datetime']):
                        if filename not in datetime_columns:
                            datetime_columns[filename] = {}
                        datetime_columns[filename][col] = (data[col].min(), data[col].max(), data[col].dt.round('T').diff().mode()[0])

            # Evaluate common time range
            first_range = None
            common_time_range = 'Yes'
            granularities = {}
            for file, cols in datetime_columns.items():
                for col, details in cols.items():
                    if first_range is None:
                        first_range = details[:2]
                    elif first_range != details[:2]:
                        common_time_range = 'No'
                    if col not in granularities:
                        granularities[col] = set()
                    granularities[col].add(details[2])
                    if len(granularities[col]) > 1:
                        granularity_issue_files.add(file)

            granularities_mismatch = {k: v for k, v in granularities.items() if len(v) > 1}
            granularity_issue_count = sum(len(v) for v in granularities_mismatch.values())

            quality_metrics = {
                'total_rows': total_rows,
                'total_null_values': total_null_values,
                'total_duplicate_rows': total_duplicate_rows,
                'duplicated_columns': list(duplicated_columns) if duplicated_columns else ['None'],
                'common_time_range': common_time_range,
                'granularity_issues': granularity_issue_files
            }
            quality, quality_score = self.evaluate_data_quality(quality_metrics)

            metrics_df = pd.DataFrame({
                'Metric': ['Total Rows', 'Total Null Values', 'Total Duplicate Rows', 'Common Time Range', 'Data Quality', 'Granularity Issue Columns', 'Duplicated Columns'],
                'Value': [total_rows, total_null_values, total_duplicate_rows, common_time_range, quality, granularity_issue_count, ', '.join(quality_metrics['duplicated_columns'])]
            })

            plt.figure(figsize=(11.69, 8.27))
            ax_table = plt.subplot(311)
            ax_table.axis('off')
            plt.title('Overall Data Quality Evaluation', fontsize=16)
            table = ax_table.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
            self.format_table_for_plot(ax_table, table)

            # Issue Table
            issue_details = []
            for issue, datasets in issue_datasets.items():
                wrapped_datasets = self.wrap_text(', '.join(datasets))
                issue_details.append([issue, wrapped_datasets])
            if granularity_issue_files:
                wrapped_granularity_files = self.wrap_text(', '.join(granularity_issue_files))
                issue_details.append(['Granularity Issues', wrapped_granularity_files])
            issue_df = pd.DataFrame(issue_details, columns=['Issue', 'Affected Datasets'])
            ax_issue_table = plt.subplot(323)
            ax_issue_table.axis('off')
            plt.title("", fontsize=12)
            issue_table = ax_issue_table.table(cellText=issue_df.values, colLabels=issue_df.columns, loc='center')
            self.format_table_for_plot(ax_issue_table, issue_table)
            for key, cell in issue_table.get_celld().items():
                cell.set_height(cell.get_height() * (len(cell.get_text().get_text().split('\n')) + 1))

            # Bar plot for Data Quality
            ax_bar_plot = plt.subplot(324)
            qualities = ['Very Bad', 'Bad', 'Normal', 'Good', 'Very Good']
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            color_dict = {'Very Bad': 'red', 'Bad': 'orange', 'Normal': 'yellow', 'Good': 'lightgreen', 'Very Good': 'green'}
            quality_color = color_dict[quality]

            ax_bar_plot.barh([''], [quality_score], color=quality_color)
            ax_bar_plot.set_xlim(0, 100)
            ax_bar_plot.set_yticks([])
            ax_bar_plot.set_xlabel('Data Quality Score')
            ax_bar_plot.set_title('Data Quality Indicator')
            ax_bar_plot.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Add legend
            handles = [plt.Rectangle((0,0),1,1, color=color_dict[q]) for q in qualities]
            ax_bar_plot.legend(handles, qualities, loc='upper right')

            pdf.savefig()
            plt.close()

            # Overall data quality summary with plots
            plt.figure(figsize=(11.69, 8.27))
            ax1 = plt.subplot(121)
            ax1.bar(range(len(null_values_stats)), null_values_stats, color='skyblue')
            ax1.set_title('Null Values Per Dataset')
            ax1.set_xlabel('Dataset Index')
            ax1.set_ylabel('Count of Null Values')

            ax2 = plt.subplot(122)
            ax2.bar(list(datatype_summary.keys()), list(datatype_summary.values()), color='lightgreen')
            ax2.set_title('Data Types Across All Datasets')
            ax2.set_xlabel('Data Type')
            ax2.set_ylabel('Count')



            pdf.savefig()
            plt.close()

        print(f"Data inspection report has been saved as '{self.report_path}'.")
    



class ZipExtractor:
    def estimate_csv_memory_usage(self, file_path, sample_size=100000):
        row_size = 0
        with open(file_path, 'r') as f:
            for _ in range(sample_size):
                line = f.readline()
                if not line:
                    break
                row_size += len(line)
        avg_row_size = row_size / sample_size if sample_size else 0
        file_size = os.path.getsize(file_path)
        num_rows = file_size / avg_row_size if avg_row_size else 0
        return int(num_rows * avg_row_size)

    def estimate_json_memory_usage(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return len(json.dumps(data).encode('utf-8'))

    def detect_file_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            return chardet.detect(f.read(10000))['encoding']

    def check_csv_integrity(self, file_path):
        try:
            pd.read_csv(file_path, nrows=10)
            return True, ""
        except Exception as e:
            return False, str(e)

    def check_json_integrity(self, file_path):
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True, ""
        except Exception as e:
            return False, str(e)

    def print_tree(self, root, file_info, level=0):
        for entry in os.scandir(root):
            if entry.is_dir():
                print(f'{"  " * level}├── {entry.name}/')
                self.print_tree(entry.path, file_info, level + 1)
            elif entry.is_file():
                file_path = entry.path
                info = file_info.get(file_path, {})
                print(f'{"  " * level}├── {entry.name} (Memory Usage: {info.get("Memory Usage", "N/A")} MB, Columns: {info.get("Columns", "N/A")})')

    def extract_zip_file(self, zip_path, password=None, show_plot=True):
        extract_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if password:
                zip_ref.setpassword(password.encode('utf-8'))
            zip_ref.extractall(extract_dir)

        file_info = {}
        total_memory_usage = 0
        large_files = []
        memory_usages = []
        file_names = []

        for root, _, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.csv'):
                    memory_usage = self.estimate_csv_memory_usage(file_path)
                    is_valid, reason = self.check_csv_integrity(file_path)
                    encoding = self.detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding) as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        num_columns = len(header)
                elif file.endswith('.json'):
                    memory_usage = self.estimate_json_memory_usage(file_path)
                    is_valid, reason = self.check_json_integrity(file_path)
                    encoding = self.detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding) as f:
                        data = json.load(f)
                        num_columns = len(data[0]) if isinstance(data, list) and data else 0
                else:
                    continue

                if not is_valid:
                    print(f'Integrity issue with {file}: {reason}')
                    continue

                total_memory_usage += memory_usage
                memory_usages.append(memory_usage)
                file_names.append(file)

                file_info[file_path] = {
                    "Memory Usage": memory_usage / (1024 ** 2),
                    "Columns": num_columns
                }

                if memory_usage > psutil.virtual_memory().available:
                    large_files.append(file_path)

        print('\nFile structure:')
        self.print_tree(extract_dir, file_info)

        print(f'\nTotal memory usage of the extracted files: {total_memory_usage / (1024 ** 2):.2f} MB')

        if large_files:
            print('\nThe following files should be processed in batches due to memory constraints:')
            for file in large_files:
                print(f'- {file}')
            batch_required = True
        else:
            print('\nAll files can be loaded into memory without batching.')
            batch_required = False

        if show_plot and memory_usages:
            largest_indices = sorted(range(len(memory_usages)), key=lambda i: memory_usages[i], reverse=True)[:3]
            largest_files = [file_names[i] for i in largest_indices]
            largest_usages = [memory_usages[i] for i in largest_indices]

            largest_usages_mb = [usage / (1024 ** 2) for usage in largest_usages]
            available_memory_mb = psutil.virtual_memory().available / (1024 ** 2)

            plt.figure(figsize=(10, 6))
            plt.bar(largest_files, largest_usages_mb, color='blue', label='Memory Required by File (MB)')
            plt.axhline(y=available_memory_mb, color='red', linestyle='-', label='Available RAM (MB)')
            plt.xlabel('Files')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage of Top 3 Largest Extracted Files')
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return not batch_required
    
    
    