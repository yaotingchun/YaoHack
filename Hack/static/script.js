function updateDetectionResults() {
    fetch('/detection_result')
        .then(response => response.json())
        .then(data => {
            document.getElementById('poseResult').textContent = data.pose;
            document.getElementById('handResult').textContent = data.hand;
            document.getElementById('scoreResult').textContent = data.score;
        })
        .catch(error => console.error('Error fetching detection results:', error));
}

setInterval(updateDetectionResults, 1000);
