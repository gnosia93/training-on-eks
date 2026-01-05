output "com_graviton_dns" {
  value       = aws_instance.graviton_box.public_dns
  description = "SSH 및 VS Code Server 접속을 위한 EC2 인스턴스의 공인 IP 주소"
}
output "com_graviton_vscode" {
    value = "http://${aws_instance.graviton_box.public_dns}:8080"
    description = "브라우저에서 VS Code 서버에 접속할 수 있는 URL (PW: 'password' by default)"
}

/*
output "com_x86_dns" {
  value       = aws_instance.x86_box.public_dns
  description = "SSH 및 VS Code Server 접속을 위한 EC2 인스턴스의 공인 IP 주소"
}
output "com_x86_vscode" {
    value = "http://${aws_instance.x86_box.public_dns}:8080"
    description = "브라우저에서 VS Code 서버에 접속할 수 있는 URL (PW: 'password' by default)"
}
*/

output "public_subnet" {
    description = "퍼블릭 서브넷 ID 목록"
    value       = aws_subnet.public[*].id
}

output "private_subnet" {
    description = "프라이빗 서브넷 ID 목록"
    value       = [for subnet in aws_subnet.private : subnet.id]
}


# 마운트에 필요한 정보 출력
output "fsx_id" {
  value       = aws_fsx_lustre_file_system.lustre_file_system.id
  description = "FSx 파일 시스템 ID입니다. (관리 및 문제 해결용)"
}

output "fsx_dns_name" {
  value       = aws_fsx_lustre_file_system.lustre_file_system.dns_name
  description = "마운트 명령어의 주소 부분에 사용됩니다."
}

output "mount_name" {
  value       = aws_fsx_lustre_file_system.lustre_file_system.mount_name
  description = "마운트 명령어의 경로 부분(마지막)에 사용됩니다."
}



