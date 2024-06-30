from flask import Flask, request, Response, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/run-script', methods=['POST'])
def run_script():
    invite_code = request.json.get('invite_code')

    # 假设这里是你的 Python 代码执行部分，输出到前端
    def generate_output():
        yield '开始执行脚本...\n'
        yield f'收到的邀请码是：{invite_code}\n'
        yield '正在处理...\n'
        yield '处理完成。\n'

    return Response(generate_output(), content_type='text/plain')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
