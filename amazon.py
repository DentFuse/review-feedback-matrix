import requests
from bs4 import BeautifulSoup
import re
import html

__all__ = ['getAmazonDescAndReviews']

# --- Config ---
URL = "https://www.amazon.in/Haier-Inverter-Window-Copper-Bacterial/dp/B0CSJNQK17"  # Replace with your target URL
reviewUrl = "https://www.amazon.in/hz/reviews-render/ajax/reviews/get/"  # Replace with your target URL
COOKIE_FILE = "cookie.txt"

# --- Load Cookie from File ---
with open(COOKIE_FILE, "r") as f:
    cookie_value = f.read().strip()

headersP = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.5",
    "Alt-Used": "www.amazon.in",
    "Connection": "keep-alive",
    "Cookie": cookie_value,
    "DNT": "1",
    "Priority": "u=0, i",
    "Referer": "https://www.amazon.in/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:137.0) Gecko/20100101 Firefox/137.0",
}

reviewPattern = r'review-text.*?<span>(.*?)</span>'

def getAmazonDescAndReviews(url):
    response = requests.get(url, headers=headersP)
    print(response)
    htmlResp = response.text
    clean_html = html.unescape(htmlResp)

    soup = BeautifulSoup(clean_html, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    descElem = soup.find(id="feature-bullets")
    desc = replaceText(descElem.get_text(strip=True) if descElem else "Description element not found")
    reviews = extractReviewsFromHtml(clean_html, False)
    return(title, desc, reviews)

def extractReviewsFromHtml(html_blob, shouldReturnResults):
    results = re.findall(reviewPattern, html_blob, re.DOTALL)
    matches = list(filter(lambda x: x != '', list(map(replaceText, list(filter(lambda x: 'report' not in x, results))))))
    if shouldReturnResults:
        return (matches, results)
    else:
        return matches

def replaceText(txt):
    textToReplace = ['\xa0', '<br />', 'About this item']
    for i in textToReplace:
        txt = txt.replace(i, '')
    return txt

# POST data as string (from --data-raw)
data = {
    "pageNumber": "1",
    "shouldAppend": "undefined",
    "deviceType": "desktop",
    "canShowIntHeader": "undefined",
    "reftag": "cm_cr_arp_d_paging_btm_next_1",
    "pageSize": "10",
    "asin": "B0BB7QGJKX",
    "scope": "reviewsAjax0"
}

def getReview(): 
    # This function requries a proper logged in csrf token set in `anti-csrftoken-a2z` in headers
    doReq = True
    pageNo = 1
    allReviews = []
    while doReq:
        print("Getting reviews page: " + str(pageNo))
        data["pageNumber"] = pageNo
        response = requests.post(reviewUrl, headers=headersP, data=data)
        html_blob = response.text
        print(response)
        (matches, results) = extractReviewsFromHtml(html_blob, True)
        allReviews += matches
        doReq = len(results) == 10
        pageNo += 1
    print(allReviews)

if __name__ == "__main__":
    # Code here runs only if this file is executed directly
    print("Running as a script")
    output = getAmazonDescAndReviews(URL)
    print(output)
else:
    # Code here runs if this file is imported as a module
    print("Imported as a module")

