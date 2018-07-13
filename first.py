# encoding=utf-8
import urllib.request
import re
from bs4 import BeautifulSoup
from distutils.filelist import findall
import json
import os

def download(url,m,n):
    os.chdir(os.path.join(os.getcwd(),"photos"))
    t = 1
    for i in range(n - m+1):
        html = urllib.request.urlopen(url + "#!s-p" + str(n+i-1))
        bs = BeautifulSoup(html,"lxml")
        for j in bs.find_all('a',class_ = 'a' ):
            mstr = str(t)+'.jpg'
            img = j.find('img').get('src')
            urllib.request.urlretrieve(img,mstr)
            print("Success!" + img)
            t += 1
        print("Next page!")

def inf_table(bs,d):
    str_list = []
    i =0
    for k in bs.find_all('table'):
        for tr in k.find_all('tr'):
            str_list.append(str(tr.get_text()).strip())
    d['table'] = str_list.copy()

def inf_infocox(bs,d):
    i =1
    for k in bs.find_all('div',class_ ='basic-info cmn-clearfix'):
        for tr in k.find_all('dt',class_ ="basicInfo-item name"):
            temp = 1
            for tt in k.find_all('dd' , class_ = "basicInfo-item value"):
                if(temp != i):
                    temp = temp + 1
                    continue
                d['infobox'][str(tr.get_text()).strip().replace('\xa0','')] = str(tt.get_text()).strip()
                break;
            i =i+1


def inf_para(bs,d):
    flag = True
    str_list = []
    for k in bs.find_all('div',class_ ='para'):
        if str(k.parent.get("class")) == "['lemma-summary']":
            flag = False
            continue
        str_list.append(str(k.get_text()).strip()+os.linesep)
    d['body'] = ''.join(str_list)

def inf_summary(bs,d):
    str_list = []
    for k in bs.find_all('div',class_ = 'lemma-summary'):
        for j in k.find_all('div',class_ = 'para'):
            str_list.append(str(j.get_text()).strip()+os.linesep)
    d['abstract'] = ''.join(str_list)

def inf_name(bs,d):
    title_node = bs.find('dd', class_="lemmaWgt-lemmaTitle-title").find("h1")
    d['name'] = str(title_node.get_text().strip())

def inf_lables(bs,d):
    str_list  = []
    for labels in bs.find_all("span", class_ ="taglist"):
        str_list.append(str(labels.get_text()).strip())
    d["labels"] = str_list

def inf_links(bs,d):
    links = {}
    for link in bs.find_all('a',href = re.compile(r"/item/")):
        links[str(link.get_text()).strip()] = str(link['href'])
    d["links"] = links


def write_json(array):
    with open("test.json", "w",encoding="utf-8") as f:
        f.write(json.dumps(array,ensure_ascii=False))
        f.close();

def get_d(url,d):
    html = urllib.request.urlopen(url)
    bs = BeautifulSoup(html, 'lxml')
    inf_name(bs,d)
    inf_summary(bs,d)
    inf_para(bs,d)
    inf_infocox(bs,d)
    inf_table(bs,d)
    inf_lables(bs,d)
    inf_links(bs, d)

d = {
    'name':"",
    'abstract':"",
    'infobox': {},
    'body':"",
    'table':[],
    'labels':[],
    'links':{}
}


array =[]
get_d("https://baike.baidu.com/item/%E6%AF%9B%E6%B3%BD%E4%B8%9C/113835",d)
with open("test.json","w",encoding="utf8") as f:
    f.write(json.dumps(d, ensure_ascii=False))

