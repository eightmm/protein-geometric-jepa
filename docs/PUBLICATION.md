# GitHub 게시 상태

## 게시 완료

[eightmm/protein-geometric-jepa](https://github.com/eightmm/protein-geometric-jepa)의 `main`에 코드, 명세, 설정, 테스트와 실행 로그가 게시되어 있습니다. 사용자가 생성한 빈 저장소를 초기화한 뒤 source import를 일반 fast-forward로 반영했습니다. 기존 `eightmm/plmol`은 변경하지 않았습니다.

| 항목 | 값 |
|---|---|
| 원본 로컬 source commit | `4fb8f95` |
| 원격 source import commit | [`2b2ee75`](https://github.com/eightmm/protein-geometric-jepa/commit/2b2ee750edd53c3d1e5b836b77d71d63b0b370e1) |
| 공통 Git tree | `dee31cbba93945cb854ef7a7ba5da39598b6997a` |
| 원본 파일 수 | 76 |
| 게시 전 CPU 재검증 | 72 passed, 3 skipped |
| 공개 범위 | 사용자가 만든 public 설정 유지 |

Import commit의 tree hash가 로컬 bundle의 tree hash와 동일하므로 경로, 내용, 실행 권한이 일치합니다. 원격의 초기화 이력이 다르므로 commit hash 자체는 원본 bundle과 다릅니다. 게시 안내 수정과 재검증 로그 추가는 이후 문서 커밋으로 분리했습니다.

[재검증 로그](../reports/publication_pytest.txt)는 로컬 CPU 결과입니다. [GitHub Actions](https://github.com/eightmm/protein-geometric-jepa/actions)는 별도 원격 실행이며, 결과를 서로 바꿔 해석하지 않습니다. CuEq/GPU 미검증 범위는 [STATUS.md](STATUS.md)에 남겨두었습니다.

## 사용

```bash
git clone https://github.com/eightmm/protein-geometric-jepa.git
cd protein-geometric-jepa
python -m pip install -e '.[dev]'
pytest -q
protein-jepa demo --config configs/smoke.yaml --output runs/smoke
```

원본 파일 스냅샷은 import commit을 checkout하여 재현할 수 있습니다. 로컬 bundle의 `v0.1.0` 태그를 원격에도 만들었다고 가정하지 마세요. 원격 source 버전의 기준은 위 import commit입니다.

## 게시 보조 스크립트

`scripts/publish_github.py`는 사용자의 인증된 `gh` 환경에서 **다른 새 private 저장소**를 만들 때 사용할 수 있습니다. 저장소 이름이 이미 존재하거나 local remote가 있으면 중단하며, force push나 기존 저장소 덮어쓰기를 수행하지 않습니다.

**이미 게시된 `eightmm/protein-geometric-jepa`에는 이 스크립트를 다시 실행할 필요가 없습니다.** 이후 변경은 정상적인 commit/push 또는 pull request로 관리하세요. 인증 토큰을 채팅이나 repository에 넣지 마세요.

## 공개 여부와 라이선스

현재 repository는 public입니다. 공개 라이선스는 소유자의 결정을 위해 설정하지 않았습니다. Public visibility와 특정 오픈소스 라이선스 부여는 별개의 결정입니다. 연구 공개 범위와 의존성 라이선스를 검토한 뒤 소유자가 라이선스를 선택할 수 있습니다.
