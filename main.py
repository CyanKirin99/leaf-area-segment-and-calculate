import cv2
import numpy as np
import os
import glob
import pandas as pd
import time
from tkinter import filedialog
from tkinter import Tk

# ==============================================================================
# 1. 全局状态变量 (Global State Variables)
# ==============================================================================
input_mode, current_input_string, blink_on, last_blink_time = None, "", True, 0
image_files, current_image_index, output_dir = [], 0, "."
rois, current_roi_points = [], []
results_data, current_sample_id, sticker_area_cm2 = [], "Demo", 8.0
hsv_thresholds = {
    'leaf': {'low': [20, 43, 30], 'high': [90, 255, 255]},
    'sticker_1': {'low': [0, 100, 100], 'high': [10, 255, 255]},
    'sticker_2': {'low': [160, 100, 100], 'high': [180, 255, 255]}
}
closing_kernel_size = 20
neutral_filter_threshold = 50

WINDOW_NAME, CONTROLS_WINDOW_NAME, PARAMS_WINDOW_NAME = "Leaf Area Analyzer V1.2", "Controls (Thresholds)", "ID & Area Controls"
MAX_DISPLAY_WIDTH, HISTORY_PANEL_WIDTH = 720, 450
INFO_PANEL_HEIGHT = 150  # V1.2: 定义信息面板的固定高度
FONT_SETTINGS = {
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'header': {'scale': 1.0, 'thick': 2},
    'large_info': {'scale': 1.0, 'thick': 2},
    'medium_info': {'scale': 0.8, 'thick': 2},
    'small_info': {'scale': 0.7, 'thick': 1}
}
params_button_coords = {
    "change_id": {"x": 10, "y": 90, "w": 430, "h": 50},
    "change_area": {"x": 10, "y": 160, "w": 430, "h": 50}
}


# ==============================================================================
# 2. 辅助函数 (Helper Functions)
# ==============================================================================
def draw_text_with_wrapping(image, text, start_pos, font_settings, max_width):
    x, y = start_pos;
    font = FONT_SETTINGS['font'];
    scale = font_settings['scale'];
    thickness = font_settings['thick']
    (text_w, text_h), _ = cv2.getTextSize("S", font, scale, thickness);
    line_height = text_h + 5
    words = text.split(' ');
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w > max_width:
            cv2.putText(image, current_line, (x, y), font, scale, (255, 255, 255), thickness)
            y += line_height;
            current_line = word
        else:
            current_line = test_line
    cv2.putText(image, current_line, (x, y), font, scale, (255, 255, 255), thickness);
    return y + line_height


def imread_utf8(path):
    try:
        stream = open(path, "rb");
        bytes_arr = bytearray(stream.read());
        numpyarray = np.asarray(bytes_arr, dtype=np.uint8)
        return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"imread_utf8 Error: {e}");
        return None


def select_folder():
    root = Tk();
    root.withdraw();
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    if folder_path: print(f"Selected: {folder_path}"); global output_dir; output_dir = folder_path
    return folder_path


def save_and_export_results():
    if not results_data:
        print("Warning: No data to export.")
        return False
    df = pd.DataFrame(results_data)
    df['sample_id'] = df['sample_id'].astype(str).apply(lambda x: f'="{x}"')
    summary = df.groupby('sample_id')['leaf_area'].sum().reset_index()
    summary = summary.rename(columns={'leaf_area': 'total_leaf_area_cm2'})
    details_path = os.path.join(output_dir, 'results_details.csv')
    summary_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(details_path, index=False, encoding='utf-8-sig')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print("\nResults exported successfully!")
    print(f"Details saved to: {details_path}")
    print(f"Summary saved to: {summary_path}")
    return True


# --- GUI回调函数 ---
def update_threshold(trackbar_name, window_name):
    key_map = {'L': 'leaf', 'S1': 'sticker_1', 'S2': 'sticker_2'}

    def _update(val):
        parts = trackbar_name.split('_');
        prefix = parts[0];
        color_key = key_map.get(prefix)
        if not color_key: return
        limit = 'low' if parts[1] == 'L' else 'high';
        ch_idx = {'H': 0, 'S': 1, 'V': 2}[parts[2]]
        hsv_thresholds[color_key][limit][ch_idx] = val

    return _update


def update_proc_params(param_name):
    def _update(val):
        global closing_kernel_size, neutral_filter_threshold
        if param_name == "closing":
            closing_kernel_size = val if val % 2 != 0 else val + 1
            if val == 0: closing_kernel_size = 0
        elif param_name == "neutral":
            neutral_filter_threshold = val

    return _update


def main_mouse_callback(event, x, y, flags, param):
    """ V1.2 MODIFIED: Handles clicks relative to the image's position below the info panel. """
    global current_roi_points, rois

    if input_mode is not None or event != cv2.EVENT_LBUTTONDOWN:
        if event == cv2.EVENT_RBUTTONDOWN:  # Allow right-click to finish ROI anytime
            if len(current_roi_points) > 2: rois.append(np.array(current_roi_points, dtype=np.int32))
            current_roi_points = []
        return

    # Extract parameters
    offset_y = param['offset_y']
    scale = param['scale']

    # Ignore clicks inside the top info panel
    if y < offset_y:
        return

    # Adjust click coordinates to be relative to the image's top-left corner
    x_on_image_display = x
    y_on_image_display = y - offset_y

    # Scale coordinates back to the original image size
    x_orig = int(x_on_image_display / scale)
    y_orig = int(y_on_image_display / scale)

    current_roi_points.append((x_orig, y_orig))


def params_mouse_callback(event, x, y, flags, param):
    global input_mode, current_input_string
    if event == cv2.EVENT_LBUTTONDOWN:
        id_btn = params_button_coords["change_id"]
        if id_btn['y'] < y < id_btn['y'] + id_btn['h'] and id_btn['x'] < x < id_btn['x'] + id_btn['w']:
            input_mode = "id";
            current_input_string = current_sample_id;
            return
        area_btn = params_button_coords["change_area"]
        if area_btn['y'] < y < area_btn['y'] + area_btn['h'] and area_btn['x'] < x < area_btn['x'] + area_btn['w']:
            input_mode = "area";
            current_input_string = str(sticker_area_cm2);
            return


# ==============================================================================
# 3. 核心图像处理与显示 (Core Image Processing and Display)
# ==============================================================================
def update_params_panel():
    global blink_on, last_blink_time
    panel = np.full((220, 450, 3), (30, 30, 30), dtype=np.uint8)
    font, white, yellow, grey = FONT_SETTINGS['font'], (255, 255, 255), (0, 255, 255), (80, 80, 80)
    cv2.putText(panel, "Click field to type, press Enter to confirm", (10, 40), font, 0.7, yellow, 1)
    cv2.line(panel, (10, 60), (440, 60), (100, 100, 100), 1)
    if time.time() - last_blink_time > 0.5: blink_on = not blink_on; last_blink_time = time.time()
    id_btn, area_btn = params_button_coords["change_id"], params_button_coords["change_area"]
    cv2.rectangle(panel, (id_btn['x'], id_btn['y']), (id_btn['x'] + id_btn['w'], id_btn['y'] + id_btn['h']), grey, -1)
    id_text = current_input_string if input_mode == "id" else current_sample_id
    if input_mode == "id" and blink_on: id_text += "|"
    cv2.putText(panel, f"ID: {id_text}", (id_btn['x'] + 15, id_btn['y'] + 35), font,
                FONT_SETTINGS['large_info']['scale'], white, FONT_SETTINGS['large_info']['thick'])
    cv2.rectangle(panel, (area_btn['x'], area_btn['y']), (area_btn['x'] + area_btn['w'], area_btn['y'] + area_btn['h']),
                  grey, -1)
    area_text = current_input_string if input_mode == "area" else f"{sticker_area_cm2:.2f}"
    if input_mode == "area" and blink_on: area_text += "|"
    cv2.putText(panel, f"Area: {area_text} cm2", (area_btn['x'] + 15, area_btn['y'] + 35), font,
                FONT_SETTINGS['large_info']['scale'], white, FONT_SETTINGS['large_info']['thick'])
    return panel


def update_history_panel(height):
    panel = np.zeros((height, HISTORY_PANEL_WIDTH, 3), dtype=np.uint8)
    font, white, yellow = FONT_SETTINGS['font'], (255, 255, 255), (0, 255, 255)
    cv2.putText(panel, "History", (10, 40), font, FONT_SETTINGS['header']['scale'], yellow,
                FONT_SETTINGS['header']['thick'])
    cv2.line(panel, (10, 60), (HISTORY_PANEL_WIDTH - 10, 60), white, 1)
    if not results_data: cv2.putText(panel, "No data saved yet.", (10, 100), font,
                                     FONT_SETTINGS['medium_info']['scale'], white,
                                     FONT_SETTINGS['medium_info']['thick']); return panel
    df = pd.DataFrame(results_data);
    summary = df.groupby('sample_id')['leaf_area'].sum()
    y_pos = 100
    for sample_id, total_area in summary.items():
        if y_pos > height - 40: cv2.putText(panel, "...", (10, y_pos), font, FONT_SETTINGS['medium_info']['scale'],
                                            white, FONT_SETTINGS['medium_info']['thick']); break
        y_pos = draw_text_with_wrapping(panel, f"{sample_id}: {total_area:.2f} cm2", (10, y_pos),
                                        FONT_SETTINGS['medium_info'], HISTORY_PANEL_WIDTH - 20)
    return panel


def process_image_and_get_data(frame):
    display_frame = frame.copy()
    calculated_data = {'leaf_px': 0, 'sticker_px': 0, 'area': 0}
    if rois:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        leaf_mask = cv2.inRange(hsv, np.array(hsv_thresholds['leaf']['low']), np.array(hsv_thresholds['leaf']['high']))
        s1 = cv2.inRange(hsv, np.array(hsv_thresholds['sticker_1']['low']),
                         np.array(hsv_thresholds['sticker_1']['high']))
        s2 = cv2.inRange(hsv, np.array(hsv_thresholds['sticker_2']['low']),
                         np.array(hsv_thresholds['sticker_2']['high']))
        sticker_mask = cv2.bitwise_or(s1, s2)

        if closing_kernel_size > 0:
            kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
            sticker_mask = cv2.morphologyEx(sticker_mask, cv2.MORPH_CLOSE, kernel)

        if neutral_filter_threshold > 0:
            neutral_mask = (cv2.absdiff(frame[:, :, 0], frame[:, :, 1]) + cv2.absdiff(frame[:, :, 0], frame[:, :,
                                                                                                      2])) <= neutral_filter_threshold
            leaf_mask[neutral_mask] = 0
            sticker_mask[neutral_mask] = 0

        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8);
        cv2.fillPoly(roi_mask, rois, 255)
        final_l_mask = cv2.bitwise_and(leaf_mask, roi_mask);
        final_s_mask = cv2.bitwise_and(sticker_mask, roi_mask)
        l_px, s_px = cv2.countNonZero(final_l_mask), cv2.countNonZero(final_s_mask)
        area = (l_px / s_px * sticker_area_cm2) if s_px > 0 else 0
        calculated_data = {'leaf_px': l_px, 'sticker_px': s_px, 'area': area}
        g_over = np.zeros_like(display_frame);
        g_over[final_l_mask == 255] = (0, 255, 0)
        r_over = np.zeros_like(display_frame);
        r_over[final_s_mask == 255] = (0, 0, 255)
        display_frame = cv2.addWeighted(display_frame, 1, g_over, 0.5, 0);
        display_frame = cv2.addWeighted(display_frame, 1, r_over, 0.5, 0)

    if rois: cv2.polylines(display_frame, rois, True, (255, 255, 0), 2)
    if len(current_roi_points) > 0:
        for p in current_roi_points: cv2.circle(display_frame, p, 5, (0, 255, 255), -1)
        if len(current_roi_points) > 1: cv2.polylines(display_frame, [np.array(current_roi_points)], False,
                                                      (0, 255, 255), 2)

    return display_frame, calculated_data


# ==============================================================================
# 4. 主程序入口 (Main Application Entry)
# ==============================================================================
def main():
    global image_files, current_image_index, rois, current_roi_points, current_sample_id, sticker_area_cm2, results_data, input_mode, current_input_string
    folder = select_folder();
    if not folder: print("No folder selected."); return
    for fmt in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'): image_files.extend(
        glob.glob(os.path.join(folder, fmt)))
    if not image_files: print(f"No images found in {folder}."); return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE);
    cv2.namedWindow(CONTROLS_WINDOW_NAME, cv2.WINDOW_NORMAL);
    cv2.namedWindow(PARAMS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, main_mouse_callback);
    cv2.setMouseCallback(PARAMS_WINDOW_NAME, params_mouse_callback)

    trackbar_prefix_map = {'Leaf': 'L', 'Sticker_1': 'S1', 'Sticker_2': 'S2'}
    for key_name, prefix in trackbar_prefix_map.items():
        val_range = 180
        for limit_char in ['L', 'H']:
            for channel_char in ['H', 'S', 'V']:
                if key_name == 'Sticker_2' and channel_char != 'H': continue
                trackbar_name = f"{prefix}_{limit_char}_{channel_char}"
                limit_val = val_range if channel_char == 'H' else 255
                val = hsv_thresholds[key_name.lower()]['low' if limit_char == 'L' else 'high']['HVS'.find(channel_char)]
                cv2.createTrackbar(trackbar_name, CONTROLS_WINDOW_NAME, val, limit_val,
                                   update_threshold(trackbar_name, CONTROLS_WINDOW_NAME))

    cv2.createTrackbar("Closing Size", CONTROLS_WINDOW_NAME, closing_kernel_size, 21, update_proc_params("closing"))
    cv2.createTrackbar("Neutral Filter", CONTROLS_WINDOW_NAME, neutral_filter_threshold, 50,
                       update_proc_params("neutral"))

    history_panel, params_panel = None, None
    while True:
        frame = imread_utf8(os.path.normpath(image_files[current_image_index]))
        if frame is None:
            if current_image_index < len(image_files) - 1:
                current_image_index += 1;
                continue
            else:
                break

        # --- V1.2: NEW LAYOUT AND SCALING LOGIC ---
        orig_h, orig_w = frame.shape[:2]

        # Calculate the available area for the image (below the info panel)
        total_display_height = int(
            (MAX_DISPLAY_WIDTH / orig_w) * orig_h)  # Maintain aspect ratio for initial height guess
        image_area_h = total_display_height - INFO_PANEL_HEIGHT
        image_area_w = MAX_DISPLAY_WIDTH

        # Calculate the correct scaling factor to fit the image in the available area
        scale_h = image_area_h / orig_h
        scale_w = image_area_w / orig_w
        scale = min(scale_h, scale_w)

        # Get the final display dimensions of the image itself
        final_img_w = int(orig_w * scale)
        final_img_h = int(orig_h * scale)

        # Update mouse callback with new parameters for accurate coordinate conversion
        cv2.setMouseCallback(WINDOW_NAME, main_mouse_callback, param={'scale': scale, 'offset_y': INFO_PANEL_HEIGHT})

        # Process the full-resolution image to get data and overlays
        processed_frame, calc_data = process_image_and_get_data(frame)

        # Resize the processed image to its final display size
        processed_frame_resized = cv2.resize(processed_frame, (final_img_w, final_img_h))

        # Create the main display area (left side of the canvas)
        # It's a black background with the size of the entire left panel
        main_display_area = np.zeros((total_display_height, MAX_DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Place the resized image onto the display area, below the info panel
        main_display_area[INFO_PANEL_HEIGHT: INFO_PANEL_HEIGHT + final_img_h, 0:final_img_w] = processed_frame_resized

        # --- Draw text onto the INFO_PANEL_HEIGHT area of the main_display_area ---
        font, white, yellow = FONT_SETTINGS['font'], (255, 255, 255), (0, 255, 255)

        img_name = f"File: {os.path.basename(image_files[current_image_index])} ({current_image_index + 1}/{len(image_files)})"
        text_end_y = draw_text_with_wrapping(main_display_area, img_name, (10, 30), FONT_SETTINGS['medium_info'],
                                             MAX_DISPLAY_WIDTH - 20)

        info2 = f"Leaf Px: {calc_data['leaf_px']} | Sticker Px: {calc_data['sticker_px']}"
        cv2.putText(main_display_area, info2, (10, text_end_y + 5), font, FONT_SETTINGS['medium_info']['scale'],
                    white, FONT_SETTINGS['medium_info']['thick'])

        info3 = f"==> Area: {calc_data['area']:.2f} cm2 <=="
        cv2.putText(main_display_area, info3, (10, text_end_y + 35), font, FONT_SETTINGS['large_info']['scale'],
                    yellow, FONT_SETTINGS['large_info']['thick'])

        controls_info = "Controls: N:Next(Save)|P:Prev|K:Skip|D:Del ROI|Z:Undo|S:Export|Q:Quit"
        cv2.putText(main_display_area, controls_info, (10, text_end_y + 65), font, 0.6, white, 1)
        # --- End of Text Drawing ---

        # Update other panels
        if history_panel is None or history_panel.shape[
            0] != total_display_height: history_panel = update_history_panel(total_display_height)
        params_panel = update_params_panel()

        # Combine all parts into the final canvas
        canvas = np.zeros((total_display_height, MAX_DISPLAY_WIDTH + HISTORY_PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[0:total_display_height, 0:MAX_DISPLAY_WIDTH] = main_display_area
        canvas[0:total_display_height, MAX_DISPLAY_WIDTH:MAX_DISPLAY_WIDTH + HISTORY_PANEL_WIDTH] = history_panel

        cv2.imshow(WINDOW_NAME, canvas);
        cv2.imshow(PARAMS_WINDOW_NAME, params_panel)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'): break
        if input_mode is not None:
            if key == 13 or key == 10:
                if input_mode == 'id':
                    if current_input_string: current_sample_id = current_input_string
                elif input_mode == 'area':
                    try:
                        sticker_area_cm2 = float(current_input_string)
                    except ValueError:
                        print(f"Invalid area: {current_input_string}")
                input_mode = None
            elif key == 8:
                current_input_string = current_input_string[:-1]
            elif 32 <= key <= 126:
                current_input_string += chr(key)
        else:  # Normal mode
            if key == ord('k'):
                if current_image_index < len(image_files) - 1: current_image_index += 1
            elif key == ord('n'):
                if rois:
                    _, final_calc_data = process_image_and_get_data(frame)
                    if final_calc_data['sticker_px'] > 0:
                        results_data.append({'sample_id': current_sample_id,
                                             'filename': os.path.basename(image_files[current_image_index]),
                                             'leaf_area': final_calc_data['area']})
                        history_panel = update_history_panel(total_display_height)
                    else:
                        print("Warning: Sticker pixels not found.")
                if current_image_index < len(image_files) - 1: current_image_index += 1
            elif key == ord('p'):
                if current_image_index > 0: current_image_index -= 1
            elif key == ord('d'):
                if rois:
                    rois.pop()
                elif current_roi_points:
                    current_roi_points = []
            elif key == ord('s'):
                if save_and_export_results(): history_panel = update_history_panel(total_display_height)
            elif key == ord('z'):
                if results_data:
                    last_entry = results_data.pop()
                    print(f"UNDO: Removed last saved entry for '{last_entry['filename']}'")
                    history_panel = update_history_panel(total_display_height)
                    if current_image_index > 0: current_image_index -= 1
                else:
                    print("UNDO: No data in memory to undo.")

    if save_and_export_results(): print("Program exited, final results saved.")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()