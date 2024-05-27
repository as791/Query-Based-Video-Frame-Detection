import falcon
from NearestFrameToContext import ProcessFramesResource
from wsgiref import simple_server
from datetime import datetime

app = falcon.App()
process_frames = ProcessFramesResource()
app.add_route('/search', process_frames)


if __name__ == '__main__':
    print(datetime.now(),"started the App")
    httpd = simple_server.make_server('localhost', 8000, app)
    httpd.serve_forever()
    