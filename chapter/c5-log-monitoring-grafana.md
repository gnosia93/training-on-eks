## [Grafana Dashboard 설정] ##

#### 1. 데이터 소스 설정 단계 ####
* Grafana 접속: 웹 브라우저에서 Grafana UI에 로그인 한다.
* Connections 메뉴 이동: 왼쪽 사이드바에서 Connections > Data sources를 클릭한다.
* 데이터 소스 추가: Add data source 버튼을 누르고 검색창에 Loki를 입력하여 선택한다.
* 상세 정보 입력 (중요):
   * Name: Loki (원하는 이름)
   * HTTP URL: Alloy 설정에 사용했던 주소의 앞부분만 입력한다.
      * 입력할 값: http://loki-gateway.loki.svc.cluster.local (주의: /loki/api/v1/push는 빼고 도메인만 입력)
   * Authenfication method 에서 Basic authentification 선택후 user / password 모두 loki 로 등록.
   * Http Headers 에서 Header 는 X-Scope-OrgId value 는 foo 입력.
* 저장 및 테스트: 하단의 Save & test 버튼을 누른다.
* Data source successfully connected.라는 초록색 메시지가 뜨면 성공.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/grafana-loki-alloy.png)

#### 2. 로그 데이터 조회 방법 (Explore) ####
* 설정이 끝났다면 실제로 로그가 나오는지 확인한다.
* 왼쪽 메뉴에서 Explore (나침반 모양 아이콘)를 클릭한다.
* 상단 드롭다운에서 방금 추가한 Loki 데이터 소스를 선택한다.
* Label browser 에서 보고자 하는 데이터를 설정한다. 
* Run query를 클릭하여 아래쪽에 실시간 로그가 올라오는지 확인한다.
![](https://github.com/gnosia93/training-on-eks/blob/main/chapter/images/loki-log.png)
