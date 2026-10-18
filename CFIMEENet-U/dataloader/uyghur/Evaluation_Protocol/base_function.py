import json
import zipfile
import re
import sys
import os
import codecs

def load_zip_file_keys(file, fileNameRegExp=''):

    try:
        archive = zipfile.ZipFile(
            file,
            mode='r',
            allowZip64=True
        )

    except:
        raise Exception(
            'Error loading the ZIP archive.'
        )

    pairs = []

    for name in archive.namelist():

        addFile = True
        keyName = name

        if fileNameRegExp != "":

            m = re.match(fileNameRegExp, name)

            if m is None:
                addFile = False

            else:
                if len(m.groups()) > 0:
                    keyName = m.group(1)

        if addFile:
            pairs.append(keyName)

    return pairs

def load_zip_file(file, fileNameRegExp='', allEntries=False):

    try:
        archive = zipfile.ZipFile(
            file,
            mode='r',
            allowZip64=True
        )

    except:
        raise Exception(
            'Error loading the ZIP archive'
        )

    pairs = []

    for name in archive.namelist():

        addFile = True
        keyName = name

        if fileNameRegExp != "":

            m = re.match(fileNameRegExp, name)

            if m is None:
                addFile = False

            else:

                if len(m.groups()) > 0:
                    keyName = m.group(1)

        if addFile:

            pairs.append([
                keyName,
                archive.read(name)
            ])

        else:

            if allEntries:

                raise Exception(
                    'ZIP entry not valid: %s' % name
                )

    return dict(pairs)


def decode_utf8(raw):


    try:

        raw = codecs.decode(
            raw,
            'utf-8',
            'replace'
        )

        raw = raw.encode('utf8')

        if raw.startswith(codecs.BOM_UTF8):

            raw = raw.replace(
                codecs.BOM_UTF8,
                '',
                1
            )

        return raw.decode('utf-8')

    except:

        return None


def validate_lines_in_file(
        fileName,
        file_contents,
        CRLF=True,
        LTRB=True,
        withTranscription=False,
        withConfidence=False,
        imWidth=0,
        imHeight=0):


    utf8File = decode_utf8(file_contents)

    if utf8File is None:

        raise Exception(
            "The file %s is not UTF-8"
            % fileName
        )

    lines = utf8File.split(
        "\r\n" if CRLF else "\n"
    )

    for line in lines:

        line = line.replace("\r", "").replace("\n", "")

        if line != "":

            try:

                validate_tl_line(
                    line,
                    LTRB,
                    withTranscription,
                    withConfidence,
                    imWidth,
                    imHeight
                )

            except Exception as e:

                raise Exception(
                    (
                        "Line in sample not valid. "
                        "Sample: %s Line: %s Error: %s"
                        % (
                            fileName,
                            line,
                            str(e)
                        )
                    ).encode('utf-8', 'replace')
                )


def validate_tl_line(
        line,
        LTRB=True,
        withTranscription=True,
        withConfidence=True,
        imWidth=0,
        imHeight=0):

    get_tl_line_values(
        line,
        LTRB,
        withTranscription,
        withConfidence,
        imWidth,
        imHeight
    )


def get_tl_line_values(
        line,
        LTRB=True,
        withTranscription=False,
        withConfidence=False,
        imWidth=0,
        imHeight=0):

    confidence = 0.0

    transcription = ""

    points = []

    numPoints = 4
    if LTRB:

        numPoints = 4

        if withTranscription and withConfidence:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect. "
                    "Should be: "
                    "xmin,ymin,xmax,ymax,confidence,transcription"
                )

        elif withConfidence:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect. "
                    "Should be: "
                    "xmin,ymin,xmax,ymax,confidence"
                )

        elif withTranscription:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect. "
                    "Should be: "
                    "xmin,ymin,xmax,ymax,transcription"
                )

        else:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect. "
                    "Should be: "
                    "xmin,ymin,xmax,ymax"
                )

        xmin = int(m.group(1))
        ymin = int(m.group(2))

        xmax = int(m.group(3))
        ymax = int(m.group(4))

        if xmax < xmin:

            raise Exception(
                "Xmax value (%s) not valid"
                % xmax
            )

        if ymax < ymin:

            raise Exception(
                "Ymax value (%s) not valid"
                % ymax
            )

        points = [
            float(m.group(i))
            for i in range(1, (numPoints + 1))
        ]

        if imWidth > 0 and imHeight > 0:

            validate_point_inside_bounds(
                xmin,
                ymin,
                imWidth,
                imHeight
            )

            validate_point_inside_bounds(
                xmax,
                ymax,
                imWidth,
                imHeight
            )

    else:

        numPoints = 8
        if withTranscription and withConfidence:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect."
                )

        elif withConfidence:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect."
                )

        elif withTranscription:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect."
                )

        else:

            m = re.match(
                r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$',
                line
            )

            if m is None:

                raise Exception(
                    "Format incorrect."
                )

        points = [
            float(m.group(i))
            for i in range(1, (numPoints + 1))
        ]

        validate_clockwise_points(points)

        if imWidth > 0 and imHeight > 0:

            validate_point_inside_bounds(
                points[0],
                points[1],
                imWidth,
                imHeight
            )

            validate_point_inside_bounds(
                points[2],
                points[3],
                imWidth,
                imHeight
            )

            validate_point_inside_bounds(
                points[4],
                points[5],
                imWidth,
                imHeight
            )

            validate_point_inside_bounds(
                points[6],
                points[7],
                imWidth,
                imHeight
            )

    # confidence
    if withConfidence:

        try:

            confidence = float(
                m.group(numPoints + 1)
            )

        except ValueError:

            raise Exception(
                "Confidence value must be a float"
            )

    if withTranscription:

        posTranscription = \
            numPoints + (2 if withConfidence else 1)

        transcription = m.group(posTranscription)

        m2 = re.match(
            r'^\s*\"(.*)\"\s*$',
            transcription
        )

        if m2 is not None:

            transcription = \
                m2.group(1) \
                .replace("\\\\", "\\") \
                .replace("\\\"", "\"")

    return points, confidence, transcription


def validate_point_inside_bounds(
        x,
        y,
        imWidth,
        imHeight):

    if x < 0 or x > imWidth:

        raise Exception(
            "X value (%s) not valid."
            % x
        )

    if y < 0 or y > imHeight:

        raise Exception(
            "Y value (%s) not valid."
            % y
        )


def validate_clockwise_points(points):

    if len(points) != 8:

        raise Exception(
            "Points list not valid."
        )

    point = [

        [int(points[0]), int(points[1])],

        [int(points[2]), int(points[3])],

        [int(points[4]), int(points[5])],

        [int(points[6]), int(points[7])]
    ]

    edge = [

        (point[1][0] - point[0][0]) *
        (point[1][1] + point[0][1]),

        (point[2][0] - point[1][0]) *
        (point[2][1] + point[1][1]),

        (point[3][0] - point[2][0]) *
        (point[3][1] + point[2][1]),

        (point[0][0] - point[3][0]) *
        (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3]

    if summatory > 0:

        raise Exception(
            "Points are not clockwise."
        )


def get_tl_line_values_from_file_contents(
        content,
        CRLF=True,
        LTRB=True,
        withTranscription=False,
        withConfidence=False,
        imWidth=0,
        imHeight=0,
        sort_by_confidences=True):

    pointsList = []

    transcriptionsList = []

    confidencesList = []

    lines = content.split(
        "\r\n" if CRLF else "\n"
    )

    for line in lines:

        line = line.replace("\r", "").replace("\n", "")

        if line != "":

            points, confidence, transcription = \
                get_tl_line_values(
                    line,
                    LTRB,
                    withTranscription,
                    withConfidence,
                    imWidth,
                    imHeight
                )

            pointsList.append(points)

            transcriptionsList.append(transcription)

            confidencesList.append(confidence)

    if (
        withConfidence
        and len(confidencesList) > 0
        and sort_by_confidences
    ):

        import numpy as np

        sorted_ind = np.argsort(
            -np.array(confidencesList)
        )

        confidencesList = [
            confidencesList[i]
            for i in sorted_ind
        ]

        pointsList = [
            pointsList[i]
            for i in sorted_ind
        ]

        transcriptionsList = [
            transcriptionsList[i]
            for i in sorted_ind
        ]

    return (
        pointsList,
        confidencesList,
        transcriptionsList
    )

def print_help():

    sys.stdout.write(
        'Usage: python %s.py -g=<gtFile> -s=<submFile> '
        '[-o=<outputFolder> -p=<jsonParams>]'
        % sys.argv[0]
    )

    sys.exit(2)


def main_evaluation(
        p,
        default_evaluation_params_fn,
        validate_data_fn,
        evaluate_method_fn,
        show_result=True,
        per_sample=True):

    if p is None:

        p = dict([
            s[1:].split('=')
            for s in sys.argv[1:]
        ])

        if len(sys.argv) < 3:
            print_help()

    evalParams = default_evaluation_params_fn()

    if 'p' in p.keys():

        evalParams.update(
            p['p']
            if isinstance(p['p'], dict)
            else json.loads(p['p'][1:-1])
        )

    resDict = {

        'calculated': True,

        'Message': '',

        'method': '{}',

        'per_sample': '{}'
    }

    try:

        validate_data_fn(
            p['g'],
            p['s'],
            evalParams
        )

        evalData = evaluate_method_fn(
            p['g'],
            p['s'],
            evalParams
        )

        resDict.update(evalData)

    except Exception as e:

        resDict['Message'] = str(e)

        resDict['calculated'] = False

    if 'o' in p:

        if not os.path.exists(p['o']):

            os.makedirs(p['o'])

        resultsOutputname = p['o'] + '/results.zip'

        outZip = zipfile.ZipFile(
            resultsOutputname,
            mode='w',
            allowZip64=True
        )

        del resDict['per_sample']

        if 'output_items' in resDict.keys():
            del resDict['output_items']

        outZip.writestr(
            'method.json',
            json.dumps(resDict)
        )

    if not resDict['calculated']:

        if show_result:

            sys.stderr.write(
                'Error!\n' +
                resDict['Message'] +
                '\n\n'
            )

        if 'o' in p:
            outZip.close()

        return resDict

    if 'o' in p:

        if per_sample is True:

            for k, v in evalData['per_sample'].iteritems():

                outZip.writestr(
                    k + '.json',
                    json.dumps(v)
                )

            if 'output_items' in evalData.keys():

                for k, v in evalData['output_items'].iteritems():

                    outZip.writestr(k, v)

        outZip.close()

    if show_result:

        sys.stdout.write("Calculated:\n")

        sys.stdout.write(
            json.dumps(resDict['method'])
        )

        print("\n")

    return resDict


def main_validation(
        default_evaluation_params_fn,
        validate_data_fn):


    try:

        p = dict([
            s[1:].split('=')
            for s in sys.argv[1:]
        ])

        evalParams = default_evaluation_params_fn()

        if 'p' in p.keys():

            evalParams.update(
                p['p']
                if isinstance(p['p'], dict)
                else json.loads(p['p'][1:-1])
            )

        validate_data_fn(
            p['g'],
            p['s'],
            evalParams
        )

        print('SUCCESS')

        sys.exit(0)

    except Exception as e:

        print(str(e))

        sys.exit(101)