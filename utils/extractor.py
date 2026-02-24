import cv2
import pytesseract
import numpy as np

def extract_form_structure(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_dominant_color(roi):
        mask = cv2.inRange(roi, np.array([0,0,0]), np.array([200,200,200]))
        if cv2.countNonZero(mask) == 0:
            return (0,0,0)
        masked_pixels = cv2.bitwise_and(roi, roi, mask=mask)
        avg_color = cv2.mean(masked_pixels, mask=mask)[:3]
        return tuple(map(int, avg_color))

    # Step 1: Extract text
    config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    raw_text_elements = []

    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text and text not in ["|", "—", "_", "-", ".", ",", "[", "]"]:
            try:
                conf = int(data['conf'][i])
            except ValueError:
                conf = -1
            if conf > 40:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                roi = img[y:y+h, x:x+w]
                avg_color = get_dominant_color(roi)
                raw_text_elements.append({"text": text, "x": x, "y": y, "w": w, "h": h, "color": avg_color})

    def clean_join(words): 
        return " ".join(words).replace("  ", " ").strip()

    def correct_text(text):
        corrections = {
            "1 Type": "ID Type", "I Type": "ID Type", "Id Type": "ID Type",
            "id Type": "ID Type", "IDtype": "ID Type", "Idtype": "ID Type",
            "Emait": "Email", "Emai": "Email", "Emal": "Email",
            "E-mail": "Email", "e-mail": "Email",
            "Addres": "Address", "Adress": "Address",
            "Signat": "Signature", "Signatur": "Signature",
            "Phon": "Phone", "Phne": "Phone",
            "Numbr": "Number", "Numb": "Number",
            "Initiat": "Initial"   # 👈 Fix OCR mistake
        }
        for wrong, right in corrections.items():
            if wrong.lower() in text.lower():
                return right
        return corrections.get(text, text)

    # Step 2: Group text into lines
    lines = []
    for el in sorted(raw_text_elements, key=lambda e: (e['y'], e['x'])):
        placed = False
        for line in lines:
            avg_h = np.mean([e['h'] for e in line['elements']])
            if abs(line['y'] - el['y']) <= avg_h * 0.8:  # relaxed grouping
                line['elements'].append(el)
                line['y'] = min(line['y'], el['y'])
                placed = True
                break
        if not placed:
            lines.append({'y': el['y'], 'elements': [el]})

    text_elements = []
    for line in lines:
        line['elements'] = sorted(line['elements'], key=lambda e: e['x'])
        current_line, prev_x = [], None
        for e in line['elements']:
            if prev_x is not None and (e['x'] - prev_x) > (e['w'] * 3):  # more tolerance
                text = clean_join([w['text'] for w in current_line])
                text = correct_text(text)
                if text:
                    avg_color = tuple(map(int, np.mean([w['color'] for w in current_line], axis=0)))
                    text_elements.append({
                        "label": text,
                        "x": min(w['x'] for w in current_line),
                        "y": min(w['y'] for w in current_line),
                        "width": max(w['x'] + w['w'] for w in current_line) - min(w['x'] for w in current_line),
                        "height": max(w['h'] for w in current_line),
                        "type": "text",
                        "color": avg_color
                    })
                current_line = []
            current_line.append(e)
            prev_x = e['x'] + e['w']
        if current_line:
            text = clean_join([w['text'] for w in current_line])
            text = correct_text(text)
            if text:
                avg_color = tuple(map(int, np.mean([w['color'] for w in current_line], axis=0)))
                text_elements.append({
                    "label": text,
                    "x": min(w['x'] for w in current_line),
                    "y": min(w['y'] for w in current_line),
                    "width": max(w['x'] + w['w'] for w in current_line) - min(w['x'] for w in current_line),
                    "height": max(w['h'] for w in current_line),
                    "type": "text",
                    "color": avg_color
                })

    # Step 3: Detect input boxes and radio buttons (unchanged)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    input_elements, radio_elements = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        roi = img[y:y+h, x:x+w]
        avg_color = get_dominant_color(roi)

        if (50 < w < 400) and (15 < h < 60):
            input_elements.append({"label": "", "x": x, "y": y, "width": w, "height": h, "type": "input", "color": avg_color})
        elif (w > 200 and 25 < h < 80):
            input_elements.append({"label": "Signature", "x": x, "y": y, "width": w, "height": h, "type": "input", "color": avg_color})
        elif (w > 200 and h >= 80):
            input_elements.append({"label": "Large Box", "x": x, "y": y, "width": w, "height": h, "type": "input", "color": avg_color})
        elif 12 < w < 30 and 12 < h < 30 and 80 < area < 900:
            aspect_ratio = w / float(h)
            if 0.75 < aspect_ratio < 1.25:
                nearby_text = [
                    t for t in text_elements 
                    if abs((t["y"] + t["height"]/2) - (y + h/2)) < 25 and abs((t["x"] + t["width"]/2) - (x + w/2)) < 120
                ]
                if nearby_text and any(kw in nearby_text[0]["label"].lower() for kw in ["type", "personal", "mortgage", "auto", "business", "purpose", "loan"]):
                    radio_elements.append({"label": nearby_text[0]["label"], "x": x, "y": y, "width": w, "height": h, "type": "radio", "color": avg_color})

    # Step 4: Attach nearest label to input boxes
    used_labels = set()
    for inp in sorted(input_elements, key=lambda b: b['y']):
        candidates = [
            t for t in text_elements
            if t['y'] + t['height'] < inp['y'] + 5
            and abs((t['x'] + t['width']/2) - (inp['x'] + inp['width']/2)) < 200  # wider tolerance
            and t['label'] not in used_labels
        ]
        if candidates:
            nearest = min(candidates, key=lambda t: inp['y'] - (t['y'] + t['height']))
            inp['label'] = nearest['label']
            used_labels.add(nearest['label'])

    elements = text_elements + input_elements + radio_elements
    return sorted(elements, key=lambda k: (k["y"], k["x"]))
