Lustre 파일 시스템을 사용하는 효율적인 방법은 완전 관리형 서비스인 Amazon FSx for Lustre 또는 FSx for Lustre CSI(Container Storage Interface) 드라이버를 활용하여 쿠버네티스 클러스터에 통합하는 것이다.

* 전제 조건
클러스터 노드(워커 노드)의 보안 그룹이 FSx 파일 시스템의 트래픽(TCP 포트 988)을 허용해야 합니다. 
