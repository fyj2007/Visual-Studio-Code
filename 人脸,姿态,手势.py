from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import os

# 加载预训练模型
model = YOLO(r"E:\Fyj20\Model\face_yolov8n.pt")


class FaceMotionTracker:
    def __init__(self, max_history=10):
        self.face_centers = deque(maxlen=max_history)
        self.motion_threshold = 5  # 运动阈值

    def track_face_motion(self, boxes):
        """跟踪人脸运动"""
        if boxes is not None and len(boxes) > 0:
            # 遍历每一个检测到的人脸框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                self.face_centers.append((center_x, center_y))

                if len(self.face_centers) > 1:
                    prev_center = self.face_centers[-2]
                    curr_center = self.face_centers[-1]

                    # 计算位移
                    dx = curr_center[0] - prev_center[0]
                    dy = curr_center[1] - prev_center[1]

                    # 判断动作类型
                    if abs(dx) > self.motion_threshold or abs(dy) > self.motion_threshold:
                        if abs(dx) > abs(dy):
                            return "左右移动", dx
                        else:
                            return "上下移动", dy
        return "静止", 0


def detect_faces(image_path):
    """检测图像中的人脸"""
    results = model(image_path)
    return results


def draw_detections(image_path, results):
    """在图像上绘制检测框"""
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'Face: {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def extract_face_roi(frame, boxes):
    """提取人脸区域"""
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]  # 取第一个人脸
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # 扩展边界框稍微大一点
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        return frame[y1:y2, x1:x2]
    return None


def compare_faces(reference_face, current_face):
    """简单的人脸比较（基于直方图相似度）"""
    if reference_face is None or current_face is None:
        return 0

    # 转换为灰度图
    ref_gray = cv2.cvtColor(reference_face, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_face, cv2.COLOR_BGR2GRAY)

    # 调整大小以匹配
    ref_resized = cv2.resize(ref_gray, (100, 100))
    curr_resized = cv2.resize(curr_gray, (100, 100))

    # 计算直方图相似度
    hist_ref = cv2.calcHist([ref_resized], [0], None, [256], [0, 256])
    hist_curr = cv2.calcHist([curr_resized], [0], None, [256], [0, 256])

    similarity = cv2.compareHist(hist_ref, hist_curr, cv2.HISTCMP_CORREL)
    return similarity


def face_verification_system():
    """人脸核对验证系统"""
    print("人脸核对验证系统")
    print("1. 请先提供参考人脸图像")
    print("2. 系统将使用摄像头扫描当前人脸")
    print("3. 进行人脸核对验证")

    # 获取参考图像路径
    reference_image_path = input("请输入参考图像路径（或输入'capture'现场拍摄）: ")

    if reference_image_path.lower() == 'capture':
        # 现场拍摄参考图像
        cap = cv2.VideoCapture(0)
        print("请将脸部对准摄像头，按下空格键拍摄参考图像...")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow('Capture Reference Face', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # 按空格键拍摄
                cv2.imwrite('reference_face.jpg', frame)
                reference_image_path = 'reference_face.jpg'
                print("参考图像已保存")
                break
            elif key == ord('q'):  # 按q键退出
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

    # 检查参考图像是否存在
    if not os.path.exists(reference_image_path):
        print(f"错误：找不到参考图像 {reference_image_path}")
        return

    # 检测参考图像中的人脸
    ref_results = detect_faces(reference_image_path)
    ref_image = cv2.imread(reference_image_path)

    # 提取参考人脸区域
    ref_boxes = None
    for result in ref_results:
        ref_boxes = result.boxes
        if ref_boxes is not None:
            ref_face = extract_face_roi(ref_image, ref_boxes)
            break

    if ref_face is None:
        print("错误：在参考图像中未检测到人脸")
        return

    print("参考人脸已提取，开始实时人脸核对验证...")
    print("按 'q' 键退出验证系统")

    # 实时人脸核对验证
    cap = cv2.VideoCapture(0)
    verification_count = 0
    match_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 人脸检测
        results = model(frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # 绘制人脸框
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face: {confidence:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                # 提取当前人脸区域
                current_face = extract_face_roi(frame, boxes)

                # 进行人脸比较
                if current_face is not None:
                    similarity = compare_faces(ref_face, current_face)
                    verification_count += 1

                    # 判断是否匹配（阈值可根据需要调整）
                    if similarity > 0.7:  # 70%相似度
                        match_result = "匹配成功"
                        match_count += 1
                        color = (0, 255, 0)  # 绿色
                    else:
                        match_result = "匹配失败"
                        color = (0, 0, 255)  # 红色

                    # 计算匹配率
                    match_rate = (match_count / verification_count * 100) if verification_count > 0 else 0

                    # 显示验证结果
                    cv2.putText(frame, f'Similarity: {similarity:.2f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    cv2.putText(frame, f'Result: {match_result}',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    cv2.putText(frame, f'Match Rate: {match_rate:.1f}%',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2)

        cv2.imshow('Face Verification System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n验证统计:")
    print(f"总验证次数: {verification_count}")
    print(f"匹配成功次数: {match_count}")
    print(f"匹配成功率: {match_rate:.1f}%")


def realtime_face_detection():
    """实时摄像头人脸检测"""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face: {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def motion_capture_detection():
    """带动作捕捉的实时人脸检测"""
    cap = cv2.VideoCapture(0)
    tracker = FaceMotionTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 人脸检测
        results = model(frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # 绘制人脸框
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face: {confidence:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                # 动作捕捉
                motion_type, motion_value = tracker.track_face_motion(boxes)
                cv2.putText(frame, f'Motion: {motion_type}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

        cv2.imshow('Face Motion Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def advanced_face_tracking():
    """高级人脸跟踪与动作捕捉"""
    cap = cv2.VideoCapture(0)
    face_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Face {i}: {confidence:.2f}',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    # 记录人脸位置用于动作分析
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    face_history.append((center_x, center_y))

                    # 保留最近10帧的位置记录
                    if len(face_history) > 10:
                        face_history.pop(0)

                    # 分析动作
                    if len(face_history) > 1:
                        dx = face_history[-1][0] - face_history[-2][0]
                        dy = face_history[-1][1] - face_history[-2][1]

                        if abs(dx) > 10:
                            direction = "右" if dx > 0 else "左"
                            cv2.putText(frame, f'水平移动: {direction}',
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 0, 0), 2)

                        if abs(dy) > 10:
                            direction = "下" if dy > 0 else "上"
                            cv2.putText(frame, f'垂直移动: {direction}',
                                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 0, 0), 2)

        cv2.imshow('Advanced Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 静态图像检测示例
    # image_path = "1.jpg"
    # results = detect_faces(image_path)
    # detected_img = draw_detections(image_path, results)
    # cv2.imwrite("1.jpg", detected_img)

    # 启动人脸核对验证系统
    face_verification_system()

    # 实时检测（取消注释以启用）
    # realtime_face_detection()
    # motion_capture_detection()
    # advanced_face_tracking()



