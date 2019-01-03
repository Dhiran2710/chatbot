from selenium import webdriver
from google_images_download import google_images_download
import os, sys
if len(sys.argv) == 1:
    print('specify download limit for each label in an argument')
downloadLimit = sys.argv[1]
response = google_images_download.googleimagesdownload()
chromedriverPath = 'C:\\Users\\Prikshet\\Desktop\\tadigital\\chromedriver.exe'
keywordsFilePath = './keywords.txt'
absolute_image_paths = response.download({'keywords_from_file':keywordsFilePath, 'limit':int(downloadLimit), 'chromedriver':chromedriverPath})
