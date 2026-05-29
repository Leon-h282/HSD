

CFG = {
    max_num_hands: 2,
    min_detection_conf: 0.7,
    min_tracking_conf: 0.5,

    frame_width: 1280,
    frame_height: 720,

    timestep: 30, // Số khung hình LSTM cần (như trong model.json)
    model_path: "./tfjs_model/tfjs_model/model.json",
    labels_path: "./LABELS/labels.json",
    stride: 5,
    moveScale: 10,

    holdTime: 2000,
};

const container = document.getElementById("camera_container");

const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");

canvasElement.width = CFG.frame_width;
canvasElement.height = CFG.frame_height;

const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    },
});

class Detector {
    constructor() {
        this.model = null;
        this.confidence = 0;
        this.probs = null;
        this.isModelLoaded = false;

        this.rawSequence = [];

        this.labels      = null;
        this.last_label  = null;
        this.label       = "_";
        this.final_label = "_";
        this.stable_count= 0;

        this.init();
    }

    async init() {
        try {
            console.log("Loading model...");
            this.model = await tf.loadLayersModel(CFG.model_path);
            this.isModelLoaded = true;
            console.log("Model loaded successfully!");

            // load labels.json
            const response = await fetch(CFG.labels_path);
            this.labels = await response.json();
            console.log("✔ Labels loaded:", this.labels);
        } catch (error) {
            console.error("Error loading model or labels:", error);
        }
    }

    async extractLandmarks(results) {
        let leftHand = new Array(63).fill(0);
        let leftWrist = new Array(3).fill(0);
        
        let rightHand = new Array(63).fill(0);
        let rightWrist = new Array(3).fill(0);

        if (
            results.multiHandLandmarks &&
            results.multiHandLandmarks.length > 0 &&
            results.multiHandedness
        ) {
            for (const [index, landmarks] of results.multiHandLandmarks.entries()) {

                let maxDist = 0;
                let normalizedList = [];

                const label = results.multiHandedness[index].label.toUpperCase();
                
                // WRIST
                const originX = landmarks[0].x;
                const originY = landmarks[0].y;
                const originZ = landmarks[0].z;

                const origin = [originX, originY, originZ];
                
                if (label === "RIGHT") {
                    rightWrist = origin;
                } else {
                    leftWrist = origin;
                }
                
                
                // HAND LANDMARKS
                let points = [];
                for (const lm of landmarks) {
                    points.push([
                        lm.x - originX,
                        lm.y - originY,
                        lm.z - originZ
                    ]);

                    const dist = Math.sqrt(
                        (lm.x - originX) ** 2
                        +
                        (lm.y - originY) ** 2
                        +
                        (lm.z - originZ) ** 2
                    );
                    if (dist > maxDist) {
                        maxDist = dist;
                    };
                };

                if (maxDist > 0) {
                    for (const p of points) {
                        normalizedList.push(
                            p[0] / maxDist,
                            p[1] / maxDist,
                            p[2] / maxDist
                        )
                    }
                } else {
                    normalizedList = points.flat()
                }

                if (label === "RIGHT") {
                    rightHand = normalizedList
                } else {
                    leftHand = normalizedList
                }
            }
        }

        return {
            leftHand: leftHand,     // Mảng 63 số
            rightHand: rightHand,   // Mảng 63 số
            leftWrist: leftWrist,   // Mảng 3 số
            rightWrist: rightWrist  // Mảng 3 số
        };
    };


    makeSequence() {
        let sequence = []

        for (let i = 0; i < this.rawSequence.length - 1; i ++) {
            const prev = this.rawSequence[i];
            const curr = this.rawSequence[i + 1];

            let left_movement  = [];
            let right_movement = [];

            for (let j = 0; j < curr.leftWrist.length; j++) {
                left_movement.push((curr.leftWrist[j] - prev.leftWrist[j]) * CFG.moveScale);
            };

            for (let j = 0; j < curr.rightWrist.length; j++) {
                right_movement.push((curr.rightWrist[j] - prev.rightWrist[j]) * CFG.moveScale);
            };

            let left_hand  = curr.leftHand.concat(left_movement);
            let right_hand = curr.rightHand.concat(right_movement);

            let frame_feature = left_hand.concat(right_hand);

            sequence.push(frame_feature);
        };

        return sequence;
    };


    async detect(results, timestep, stride) {
        let currentData;

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            currentData = await this.extractLandmarks(results);
        }
        else {
            currentData = {
                leftHand: new Array(63).fill(0),
                rightHand: new Array(63).fill(0),
                leftWrist: new Array(3).fill(0),
                rightWrist: new Array(3).fill(0)
            };
        }
        
        this.rawSequence.push(currentData);

        if (this.rawSequence.length > timestep + 1) {
            this.rawSequence.splice(0, stride + 1);
        };

        if (this.isModelLoaded && this.rawSequence.length === timestep + 1) {
            let sequence = this.makeSequence();
            await this.predict(sequence);
        };

        return this.final_label;
    };


    async predict(sequence) {
        if (!this.isModelLoaded) return;

        // 3. Tiến hành dự đoán khi đã tích lũy đủ 30 khung hình liên tiếp
        if (this.rawSequence.length === CFG.timestep + 1) {

            const inputTensor = tf.tensor3d([sequence]);
            const prediction = this.model.predict(inputTensor);
            
            // Lấy mảng xác suất (probability) đầu ra
            const scores = await prediction.data();
            
            // Tìm index có xác suất cao nhất (Tương đương np.argmax)
            const maxIndex = prediction.argMax(-1).dataSync()[0];
            const maxScore = scores[maxIndex];

            // Giải phóng bộ nhớ Tensor để tránh tràn RAM trình duyệt
            inputTensor.dispose();
            prediction.dispose();

            // 4. Nếu độ tự tin > 75%, hiển thị kết quả ra màn hình
            if (maxScore > 0.75 && this.labels) {
                this.label = this.labels[maxIndex];
                // console.log(`Prediction: ${actionLabel} (${(maxScore * 100).toFixed(2)}%)`);
            } else {
                this.label = "_";
            }

            if (this.label !== this.last_label) {
                this.stable_count = Date.now() + CFG.holdTime;
                this.last_label = this.label;
                this.final_label = "_"
                this.rawSequence = [];
            }

            if (Date.now() > this.stable_count) {
                this.final_label = this.label
            }
            return this.final_label;
        }
    }

    async rightOnFrame() {
        const predResultElement = document.getElementById("text");
        predResultElement.innerText = await this.final_label;
    }
}


// KHỞI TẠP DETECTOR
const detector = new Detector();


// MEDIAPIPE HAND
hands.setOptions({
    maxNumHands: CFG.max_num_hands,
    minDetectionConfidence: CFG.min_detection_conf,
    minTrackingConfidence: CFG.min_tracking_conf,
    modelComplexity: 1,
});

// CHECK HAND IN FRAME
function checkHandInFrame(results) {
    const handStatusElement = document.getElementById("handInFrame");

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        handStatusElement.innerText = "Hand In Frame: Yes";
        handStatusElement.style.color = "#02d140";
    } else {
        handStatusElement.innerText = "Hand In Frame: No";
        handStatusElement.style.color = "#d10202";
    }
}

// DRAW HAND LANDMARKS
function drawHandLandmarks(results) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Draw Hand Landmarks
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(
                canvasCtx,
                landmarks,
                HAND_CONNECTIONS
            );

            drawLandmarks(
                canvasCtx,
                landmarks
            );
        }
    }
}

// Get mediapipe hand results
function onResults(results) {
    console.log("results received");

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // FLIP FRAME
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);

    canvasCtx.drawImage(
        results.image,
        0, 0,
        canvasElement.width,
        canvasElement.height,
    );

    drawHandLandmarks(results);

    canvasCtx.restore();
    checkHandInFrame(results);

    detector.detect(results, CFG.timestep, CFG.stride);
    detector.rightOnFrame();
}

hands.onResults(onResults);

// CAMERA
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({
        image: videoElement,
        });
    },

    width: CFG.frame_width,
    height: CFG.frame_height,
});

camera.start();