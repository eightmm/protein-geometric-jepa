# GitHub 게시

## 현재 상태

`protein-geometric-jepa`의 코드/명세/테스트는 로컬 Git 프로젝트로 작성되었습니다. 원격 GitHub repository는 아직 만들지 않았습니다. 이 대화의 연결 도구에는 기존 repository 쓰기는 있지만 새 repository 생성은 없습니다. 기존 `eightmm/plmol`에는 어떤 변경도 적용하지 않았습니다.

## 방법 A — 인증된 본인 컴퓨터에서 게시

소스 ZIP을 풀고 프로젝트 최상위 폴더에서 아래를 실행합니다. Git과 GitHub CLI가 필요합니다. 토큰을 채팅에 보내지 말고 본인 컴퓨터에서 인증하세요.

```bash
gh auth login
python scripts/publish_github.py --dry-run
python scripts/publish_github.py --repo eightmm/protein-geometric-jepa
```

이 스크립트는 다음 제한을 갖습니다.

- 자신의 authenticated GitHub account에 **새 private repository만** 생성합니다.
- 이름이 이미 존재하면 멈춥니다. 기존 저장소를 재사용하거나 force push하지 않습니다.
- 기존 local remote가 있으면 멈추며 설정을 바꾸지 않습니다.
- ZIP에 `.git`이 없으면 `main`으로 초기화하고 현재 소스를 commit합니다.
- Global git 설정을 바꾸지 않습니다.
- Dataset, checkpoint, `.env`는 `.gitignore` 대상입니다. 게시 전 `git status`와 파일을 직접 검토하세요.

GitHub API 정책, 계정 권한, 네트워크 오류가 발생하면 생성 또는 push가 실패할 수 있습니다. 생성은 됐지만 push만 실패했다면 반복 실행은 기존 repository 감지로 중단합니다. 그때는 해당 remote와 오류를 직접 확인한 뒤 일반 `git push -u origin main`을 사용하세요. 스크립트가 실패한 원격 repository를 삭제하지 않습니다.

## 방법 B — GitHub에서 빈 저장소를 만든 뒤 이 대화로 연결

GitHub에서 `eightmm/protein-geometric-jepa`를 **private**, **README 초기화 포함**으로 만듭니다. GitHub connector가 선택된 repository만 허용한다면 새 repository도 접근 대상으로 추가합니다. 그 URL을 이 대화에 전달하면 기존-repository write action으로 코드와 문서를 게시하는 후속 작업이 가능합니다.

README 초기화를 권하는 이유는 현재 connector의 commit action이 parent commit을 요구하기 때문입니다. 빈 Git history를 추정해서 임의로 조작하지 않습니다.

## Git bundle 사용

Git bundle은 commit history와 refs를 담은 portable local Git archive입니다. GitHub 자체를 의미하지 않습니다.

```bash
git clone protein-geometric-jepa-v0.1.0.bundle protein-geometric-jepa
cd protein-geometric-jepa
git log --oneline
```

bundle에서 clone하면 origin이 local bundle 경로일 수 있습니다. `git remote -v`를 확인하세요. 게시 스크립트는 기존 remote를 자동 삭제하지 않습니다. 처음 게시에는 ZIP을 풀어서 방법 A를 실행하는 것이 단순합니다.

## 공개 여부와 라이선스

기본 게시 모드는 private입니다. 본 프로젝트의 공개 라이선스는 사용자가 결정하도록 남겨두었습니다. 이 배포가 자동으로 공개 오픈소스 라이선스를 부여하는 것은 아닙니다. 공개 전 연구 공개 범위와 사용하는 dependency의 라이선스를 검토하세요.
