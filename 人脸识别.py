import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
import time
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO


class FaceHandMotionSystem:
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化人脸识别和手部动作识别系统

        Args:
            config: 系统配置参数
        """
        self.config = config or self._get_default_config()
        self._initialize_models()
        # 添加手势历史记录用于稳定识别
        self.gesture_history = {}
        self.history_length = 5

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'face_detection': {
                'model_path': r'E:\Fyj20\Model\face_yolov8n.pt',
                'confidence': 0.3,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'hand_detection': {
                'model_path': r'E:\Fyj20\Model\hand_yolov8n.pt',
                'confidence': 0.4,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'person_detection': {
                'model_path': r'E:\Fyj20\Model\person_yolov8n-seg.pt',
                'confidence': 0.5
            },
            'mediapipe': {
                'max_num_faces': 5,
                'max_num_hands': 2,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.7
            }
        }

    def _initialize_models(self):
        """初始化模型"""
        try:
            # 直接使用YOLO加载人脸检测模型
            self.face_model = YOLO(self.config['face_detection']['model_path'])

            # 直接使用YOLO加载手部检测模型
            self.hand_model = YOLO(self.config['hand_detection']['model_path'])

            # 初始化YOLO人物检测模型
            self.person_model = YOLO(self.config['person_detection']['model_path'])

            # 初始化MediaPipe
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            # 创建MediaPipe实例
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.config['mediapipe']['max_num_faces'],
                refine_landmarks=True,
                min_detection_confidence=self.config['mediapipe']['min_detection_confidence'],
                min_tracking_confidence=self.config['mediapipe']['min_tracking_confidence']
            )

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config['mediapipe']['max_num_hands'],
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            print("系统初始化完成，人脸和手部检测模型已加载")
            print(f"人物分割模型已加载: {self.config['person_detection']['model_path']}")
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的人脸"""
        try:
            results = self.face_model(image, conf=self.config['face_detection']['confidence'])
            faces = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        face_data = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'expressions': {}
                        }
                        faces.append(face_data)

            return faces
        except Exception as e:
            print(f"人脸检测出错: {e}")
            return []

    def detect_hands(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的手部并识别手势"""
        try:
            results = self.hand_model(image, conf=self.config['hand_detection']['confidence'])
            hands = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # 使用 MediaPipe 进行精细的手势识别
                        hand_roi = image[y1:y2, x1:x2]
                        gesture = "Unknown"
                        handedness = "Unknown"

                        if hand_roi.size > 0:
                            rgb_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                            mp_results = self.hands.process(rgb_hand)

                            if mp_results.multi_hand_landmarks:
                                # 手势识别
                                gesture = self._recognize_gesture(mp_results.multi_hand_landmarks[0])

                                # 使用历史记录稳定手势识别
                                hand_id = f"{x1}_{y1}_{x2}_{y2}"
                                gesture = self._stabilize_gesture(hand_id, gesture)

                                # 确定左手还是右手
                                if len(mp_results.multi_handedness) > 0:
                                    handedness = "Right" if mp_results.multi_handedness[0].classification[
                                                                0].index == 1 else "Left"

                        hand_data = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'gesture': gesture,
                            'handedness': handedness
                        }
                        hands.append(hand_data)

            return hands
        except Exception as e:
            print(f"手部检测出错: {e}")
            return []

    def _stabilize_gesture(self, hand_id: str, gesture: str) -> str:
        """通过历史记录稳定手势识别结果"""
        if hand_id not in self.gesture_history:
            self.gesture_history[hand_id] = []

        self.gesture_history[hand_id].append(gesture)

        # 保持历史记录长度
        if len(self.gesture_history[hand_id]) > self.history_length:
            self.gesture_history[hand_id].pop(0)

        # 返回出现频率最高的手势
        if self.gesture_history[hand_id]:
            most_common = max(set(self.gesture_history[hand_id]),
                              key=self.gesture_history[hand_id].count)
            return most_common

        return gesture

    def _recognize_gesture(self, hand_landmarks) -> str:
        """基于手部关键点识别手势"""
        if not hand_landmarks:
            return "Unknown"

        landmarks = hand_landmarks.landmark

        # 计算手指弯曲程度
        finger_states = self._calculate_finger_states(landmarks)
        extended_fingers = sum(finger_states)

        # 更精确的手势识别
        if extended_fingers == 0:
            return "Fist"
        elif extended_fingers == 5:
            return "Open_Palm"
        elif extended_fingers == 1 and finger_states[1]:  # 食指
            return "Pointing"
        elif extended_fingers == 1 and finger_states[0]:  # 拇指
            return "Thumbs_Up"
        elif extended_fingers == 2 and finger_states[1] and finger_states[2]:  # 食指和中指
            return "Victory"
        elif extended_fingers == 4 and not finger_states[0]:  # 除拇指外四指
            return "Four_Fingers"
        else:
            return "Other_Gesture"

    def _calculate_finger_states(self, landmarks) -> List[bool]:
        """计算每个手指的弯曲状态"""
        # 手指索引: [拇指, 食指, 中指, 无名指, 小指]
        finger_states = []

        # 拇指 (比较拇指尖和拇指第二个关节与手腕的位置)
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]

        # 计算向量方向判断拇指是否伸直
        thumb_vector = np.array([thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y])
        palm_vector = np.array([wrist.x - thumb_mcp.x, wrist.y - thumb_mcp.y])
        thumb_angle = np.arccos(np.dot(thumb_vector, palm_vector) /
                                (np.linalg.norm(thumb_vector) * np.linalg.norm(palm_vector)))
        finger_states.append(thumb_angle > 0.5)

        # 其他手指 (比较指尖和第一个关节与手掌根部的位置)
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        wrist_coords = np.array([wrist.x, wrist.y])

        for tip, pip in zip(finger_tips, finger_pips):
            tip_coords = np.array([landmarks[tip].x, landmarks[tip].y])
            pip_coords = np.array([landmarks[pip].x, landmarks[pip].y])

            # 计算相对于手掌根部的距离
            tip_dist = np.linalg.norm(tip_coords - wrist_coords)
            pip_dist = np.linalg.norm(pip_coords - wrist_coords)

            finger_states.append(tip_dist > pip_dist)

        return finger_states

    def detect_persons(self, image: np.ndarray) -> List[Dict]:
        """检测图像中的人物并返回分割掩码"""
        try:
            results = self.person_model(image, conf=self.config['person_detection']['confidence'])
            persons = []

            for i, result in enumerate(results):
                if result.boxes is not None and len(result.boxes) > 0:
                    for j, box in enumerate(result.boxes):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])

                        # 如果有分割掩码，则获取掩码数据
                        mask = None
                        if result.masks is not None and len(result.masks) > j:
                            mask = result.masks[j].data.cpu().numpy()

                        person_data = {
                            'id': len(persons),
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'mask': mask,
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        persons.append(person_data)

            return persons
        except Exception as e:
            print(f"人物检测出错: {e}")
            return []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧图像，检测人脸、手部和人物"""
        try:
            # 检测人脸
            faces = self.detect_faces(frame)

            # 检测手部
            hands = self.detect_hands(frame)

            # 检测人物（包括分割）
            persons = self.detect_persons(frame)

            # 绘制人物检测结果
            for person in persons:
                bbox = person['bbox']
                confidence = person['confidence']

                # 绘制人物边界框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, f"Person: {confidence:.2f}",
                            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 如果有人物分割掩码，则叠加显示
                if person['mask'] is not None and len(person['mask']) > 0:
                    try:
                        # 调整掩码大小以匹配图像尺寸
                        mask_resized = cv2.resize(person['mask'][0], (frame.shape[1], frame.shape[0]))
                        # 创建彩色掩码
                        color_mask = np.zeros_like(frame)
                        color_mask[:, :] = [0, 0, 255]  # 红色掩码
                        # 应用掩码
                        masked_img = np.where(mask_resized[..., None] > 0.5, color_mask, frame)
                        # 混合原图和掩码
                        frame = cv2.addWeighted(frame, 0.7, masked_img, 0.3, 0)
                    except Exception as e:
                        print(f"掩码处理出错: {e}")

            # 绘制人脸检测结果
            for face in faces:
                bbox = face['bbox']
                confidence = face['confidence']
                expressions = face['expressions']

                # 绘制人脸边界框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {confidence:.2f}",
                            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 显示表情信息
                if 'expression' in expressions:
                    cv2.putText(frame, f"Expression: {expressions['expression']}",
                                (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制手部检测结果
            for hand in hands:
                bbox = hand['bbox']
                confidence = hand['confidence']
                gesture = hand['gesture']
                handedness = hand['handedness']

                # 绘制手部边界框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"{handedness} Hand: {confidence:.2f}",
                            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 显示手势信息
                cv2.putText(frame, f"Gesture: {gesture}",
                            (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            return frame
        except Exception as e:
            print(f"帧处理出错: {e}")
            return frame


# 使用示例
if __name__ == "__main__":
    try:
        # 创建系统实例
        system = FaceHandMotionSystem()

        # 打开摄像头
        cap = cv2.VideoCapture(0)

        # 设置摄像头参数以提高性能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("系统启动成功，按 'q' 键退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break

            # 处理帧
            processed_frame = system.process_frame(frame)

            # 显示结果
            cv2.imshow('Face Hand Motion System', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 清理资源
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
