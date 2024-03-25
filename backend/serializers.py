# serializers.py
from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000)
    answer = serializers.CharField(max_length=1000)