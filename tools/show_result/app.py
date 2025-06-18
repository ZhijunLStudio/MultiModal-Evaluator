from flask import Flask, render_template, request, redirect, url_for
import os
import glob

app = Flask(__name__)

RESULTS_DIR = '../../.cache/results'

def get_html_files():
    """获取所有HTML文件并按序号排序"""
    files = glob.glob(os.path.join(RESULTS_DIR, '*.html'))
    return sorted(files, key=lambda x: int(os.path.basename(x).split('_')[0]))

def get_file_index(filename):
    """获取文件在列表中的索引"""
    files = get_html_files()
    try:
        return files.index(filename)
    except ValueError:
        return -1

@app.route('/')
def index():
    """主页，显示所有可用的HTML文件"""
    files = get_html_files()
    file_list = []
    for file in files:
        basename = os.path.basename(file)
        number = basename.split('_')[0]
        file_list.append({
            'number': number,
            'name': basename,
            'path': file
        })
    return render_template('index.html', files=file_list)

@app.route('/view/<path:filename>')
def view_file(filename):
    """查看指定的HTML文件"""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    files = get_html_files()
    current_index = get_file_index(filepath)
    
    prev_file = files[current_index - 1] if current_index > 0 else None
    next_file = files[current_index + 1] if current_index < len(files) - 1 else None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return render_template('viewer.html',
                         content=content,
                         current_file=filename,
                         prev_file=os.path.basename(prev_file) if prev_file else None,
                         next_file=os.path.basename(next_file) if next_file else None)

@app.route('/search')
def search():
    """搜索文件"""
    query = request.args.get('query', '').strip()
    if not query:
        return redirect(url_for('index'))
    
    files = get_html_files()
    results = []
    
    for file in files:
        basename = os.path.basename(file)
        if query in basename:
            number = basename.split('_')[0]
            results.append({
                'number': number,
                'name': basename,
                'path': file
            })
    
    return render_template('index.html', files=results, search_query=query)

if __name__ == '__main__':
    app.run(debug=True, host='192.168.99.119', port=5000) 