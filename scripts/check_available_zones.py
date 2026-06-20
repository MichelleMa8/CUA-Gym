#!/usr/bin/env python3
"""Query Alibaba Cloud for regions/zones where ecs.g8a.xlarge is available."""
import os
import sys
from alibabacloud_ecs20140526.client import Client as EcsClient
from alibabacloud_tea_openapi.models import Config
from alibabacloud_ecs20140526.models import DescribeAvailableResourceRequest

INSTANCE_TYPE = os.getenv("ALIYUN_INSTANCE_TYPE", "ecs.g8a.xlarge")

CN_REGIONS = [
    "cn-hangzhou", "cn-shanghai", "cn-beijing", "cn-shenzhen",
    "cn-chengdu", "cn-zhangjiakou", "cn-huhehaote", "cn-wulanchabu",
    "cn-nanjing", "cn-fuzhou", "cn-guangzhou",
]


def make_client(region: str) -> EcsClient:
    config = Config(
        access_key_id=os.environ["ALIYUN_ACCESS_KEY_ID"],
        access_key_secret=os.environ["ALIYUN_ACCESS_KEY_SECRET"],
        region_id=region,
        endpoint=f"ecs.{region}.aliyuncs.com",
    )
    return EcsClient(config)


def check_region(region: str) -> list[tuple[str, str]]:
    try:
        client = make_client(region)
        req = DescribeAvailableResourceRequest(
            region_id=region,
            destination_resource="InstanceType",
            instance_type=INSTANCE_TYPE,
        )
        resp = client.describe_available_resource(req)
        results = []
        for az in resp.body.available_zones.available_zone:
            for res in (az.available_resources.available_resource or []):
                for info in (res.supported_resources.supported_resource or []):
                    if info.value == INSTANCE_TYPE and info.status_category == "WithStock":
                        results.append((az.zone_id, info.status_category))
        return results
    except Exception as e:
        err = str(e)
        if "InvalidRegionId" in err or "Forbidden" in err:
            return []
        print(f"  [{region}] error: {err[:120]}", file=sys.stderr)
        return []


def main():
    for k in ("ALIYUN_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_SECRET"):
        if k not in os.environ:
            sys.exit(f"Missing env var: {k}")

    print(f"Checking availability of {INSTANCE_TYPE} across {len(CN_REGIONS)} regions...\n")
    available = []
    for region in CN_REGIONS:
        zones = check_region(region)
        if zones:
            for zone_id, status in zones:
                print(f"  AVAILABLE  {region}  zone={zone_id}  status={status}")
                available.append((region, zone_id))
        else:
            print(f"  no stock    {region}")

    print(f"\n{'='*60}")
    if available:
        print(f"Found {len(available)} zone(s) with stock:")
        for region, zone in available:
            print(f"  ALIYUN_REGION={region}  (zone: {zone})")
        print("\nTo use the first available region, set in your .env:")
        r, z = available[0]
        print(f"  ALIYUN_REGION={r}")
    else:
        print("No zones found with stock. Consider trying a different instance type.")


if __name__ == "__main__":
    main()
