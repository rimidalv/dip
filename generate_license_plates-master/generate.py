import sys, os
alphabet = u'ABCEHKMOPTXY'
nums = u'0123456789'

import random as rn

def get_abc():
    return alphabet[rn.randint(0, len(alphabet)-1)]

def gen_string_num():
    num1 = get_abc()
    num2 = get_abc()
    num3 = get_abc()
    dig1 = rn.randint(0, 9)
    dig2 = rn.randint(0, 9)
    dig3 = rn.randint(0, 9)

    numFin = rn.randint(1, 99)
    return "{}{}{}{}{}{}{:02d}".format(num1, dig1, dig2, dig3, num2, num3, numFin )



def svg2png(svgcode, fname='output.png'):
    import cairosvg
    # with open(fname, 'wb') as fout:
    data = cairosvg.svg2png(bytestring=svgcode)
    return fname, data    

def set_numbers(svg, numbers='m976mm34'):
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(svg)

    for elem in tree.iter('{http://www.w3.org/2000/svg}text'):
        id = elem.attrib['id']
        if id.startswith('plate'):
            
            text = ''
            for c in id[5:]:
                text += numbers[int(c)]
            elem[0].text = text
    return ET.tostring(tree)


def generateNumber():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-o", '--output')
    # parser.add_argument('number')
    # args = parser.parse_args()

    # pngout = args.number + '.png'
    # if args.output:
    #     pngout = args.output
    # print(pngout)
    label = gen_string_num()
    # print(label)
    code = open(os.path.join(os.path.dirname(__file__), 'ru.svg'), 'r').read()
    code = set_numbers(code, numbers=label)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output.png'))
    imageName, data = svg2png(code, fname=out_path)
    return (data, label)


