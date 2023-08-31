from concurrent import futures
import grpc
from proto import predict_pb2
from proto import predict_pb2_grpc
from model_operations import load_model

class PredictionServicer(predict_pb2_grpc.PredictionServicer):
    def GetPrediction(self, request, context):
        text = request.text
        model = load_model('model.pickle')
        vectorizer = load_model('vectorizer.pickle')
        tfidf_vector = vectorizer.transform([text])
        result = model.predict_proba(tfidf_vector)

        prediction = "Confirmation_Yes" if result[0][0] > result[0][1] else "Confirmation_No"
        return predict_pb2.PredictionResponse(prediction=prediction)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    predict_pb2_grpc.add_PredictionServicer_to_server(PredictionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
